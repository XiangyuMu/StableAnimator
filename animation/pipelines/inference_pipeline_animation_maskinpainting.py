import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import PIL.Image
import einops
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor, PipelineImageInput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion \
    import _resize_with_antialiasing, _append_dims
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor

from animation.modules.attention_processor import AnimationAttnProcessor, AnimationIDAttnProcessor
from einops import rearrange

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def tensor2vid(video: torch.Tensor, processor: "VaeImageProcessor", output_type: str = "np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)
        outputs.append(batch_output)

    return outputs
def load_model_to_device(model, device, dtype=None):
    """将模型加载到指定设备和数据类型"""
    model.to(device)
    if dtype is not None:
        model = model.to(dtype)
    return model

@dataclass
class InferenceAnimationPipelineOutput(BaseOutput):
    r"""
    Output class for mimicmotion pipeline.

    Args:
        frames (`[List[List[PIL.Image.Image]]`, `np.ndarray`, `torch.Tensor`]):
            List of denoised PIL images of length `batch_size` or numpy array or torch tensor of shape `(batch_size,
            num_frames, height, width, num_channels)`.
    """

    frames: Union[List[List[PIL.Image.Image]], np.ndarray, torch.Tensor]


class InferenceAnimationPipeline(DiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKLTemporalDecoder`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K]
            (https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNetSpatioTemporalConditionModel`]):
            A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
        scheduler ([`EulerDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.
        pose_net ([`PoseNet`]):
            A `` to inject pose signals into unet.
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
            self,
            vae,
            image_encoder,
            unet,
            scheduler,
            feature_extractor,
            pose_net,
            face_encoder,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            pose_net=pose_net,
            face_encoder=face_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.num_tokens = 4

        # self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # self.app.prepare(ctx_id=0, det_size=(640, 640))
        # self.lora_rank = 128
        # self.set_ip_adapter()

    def get_prepare_faceid(self, face_image):
        faceid_image = np.array(face_image)
        faces = self.app.get(faceid_image)
        if faces == []:
            faceid_embeds = torch.zeros_like(torch.empty((1, 512)))
        else:
            faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        return faceid_embeds

    def set_ip_adapter(self):
        unet = self.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AnimationAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=self.lora_rank,
                ).to(self.device, dtype=self.torch_dtype)
            else:
                attn_procs[name] = AnimationIDAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0, rank=self.lora_rank,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=self.torch_dtype)

        unet.set_attn_processor(attn_procs)

    def _encode_image(
            self,
            image: PipelineImageInput,
            device: Union[str, torch.device],
            num_videos_per_prompt: int,
            do_classifier_free_guidance: bool):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

            # Normalize the image with for CLIP input
            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    def _encode_vae_image(
            self,
            image: torch.Tensor,
            device: Union[str, torch.device],
            num_videos_per_prompt: int,
            do_classifier_free_guidance: bool,
    ):
        image = image.to(device=device, dtype=self.vae.dtype)
        image_latents = self.vae.encode(image).latent_dist.mode()

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents

    def _get_add_time_ids(
            self,
            fps: int,
            motion_bucket_id: int,
            noise_aug_strength: float,
            dtype: torch.dtype,
            batch_size: int,
            num_videos_per_prompt: int,
            do_classifier_free_guidance: bool,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, " \
                f"but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. " \
                f"Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    def decode_latents(
            self,
            latents: torch.Tensor,
            num_frames: int,
            decode_chunk_size: int = 8):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i: i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i: i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame.cpu())
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    def check_inputs(self, image, height, width):
        if (
                not isinstance(image, torch.Tensor)
                and not isinstance(image, PIL.Image.Image)
                and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(
            self,
            batch_size: int,
            num_frames: int,
            num_channels_latents: int,
            height: int,
            width: int,
            dtype: torch.dtype,
            device: Union[str, torch.device],
            generator: torch.Generator,
            latents: Optional[torch.Tensor] = None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        if isinstance(self.guidance_scale, (int, float)):
            return self.guidance_scale > 1
        return self.guidance_scale.max() > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
            self,
            image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
            image_pose: Union[torch.FloatTensor],
            masks: Union[torch.FloatTensor], #掩码
            original_images: Union[torch.FloatTensor], #原始图像
            height: int = 576,
            width: int = 1024,
            num_frames: Optional[int] = None,
            tile_size: Optional[int] = 16,
            tile_overlap: Optional[int] = 4,
            num_inference_steps: int = 25,
            min_guidance_scale: float = 1.0,
            max_guidance_scale: float = 3.0,
            fps: int = 7,
            motion_bucket_id: int = 127,
            noise_aug_strength: float = 0.02,
            image_only_indicator: bool = False,
            decode_chunk_size: Optional[int] = None,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            validation_image_id_ante_embedding=None,
            output_type: Optional[str] = "pil",
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            return_dict: bool = True,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/
                feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid`
                and to 25 for `stable-video-diffusion-img2vid-xt`
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second.The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation.
                The higher the number the more motion will be in the video.
            noise_aug_strength (`float`, *optional*, defaults to 0.02):
                The amount of noise added to the init image,
                the higher it is the less the video will look like the init image. Increase it for more motion.
            image_only_indicator (`bool`, *optional*, defaults to False):
                Whether to treat the inputs as batch of images instead of videos.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time.The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption.
                By default, the decoder will decode all frames at once for maximal quality.
                Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            device:
                On which device the pipeline runs on.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`,
                [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image(
        "https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        ```
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = max_guidance_scale >= 1.0
        self._guidance_scale = max_guidance_scale

        # 3. Encode input image
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)
        # self.image_encoder.cpu()

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        fps = fps - 1

        # 4. Encode input image using VAE

        # print(image_embeddings.size()) # [2, 1, 1024]
        validation_image_id_ante_embedding = torch.from_numpy(validation_image_id_ante_embedding).unsqueeze(0)
        validation_image_id_ante_embedding = validation_image_id_ante_embedding.to(device=device, dtype=image_embeddings.dtype)

        faceid_latents = self.face_encoder(validation_image_id_ante_embedding, image_embeddings[1:])
        # print(faceid_latents.size()) # [1, 4, 1024]
        uncond_image_embeddings = image_embeddings[:1]
        uncond_faceid_latents = torch.zeros_like(faceid_latents)
        uncond_image_embeddings = torch.cat([uncond_image_embeddings, uncond_faceid_latents], dim=1)
        cond_image_embeddings = image_embeddings[1:]
        cond_image_embeddings = torch.cat([cond_image_embeddings, faceid_latents], dim=1)
        image_embeddings = torch.cat([uncond_image_embeddings, cond_image_embeddings])

        image = self.image_processor.preprocess(image, height=height, width=width).to(device)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = (self.vae.dtype == torch.float16 or self.vae.dtype == torch.bfloat16) and self.vae.config.force_upcast
        if needs_upcasting:
            self_vae_dtype = self.vae.dtype
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        image_latents = image_latents.to(image_embeddings.dtype)

        if needs_upcasting:
            self.vae.to(dtype=self_vae_dtype)
        # self.vae.cpu()

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            tile_size,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )
        latents = latents.repeat(1, num_frames // tile_size + 1, 1, 1, 1)[:, :num_frames]
        # 处理original_images
        original_images_list = []
        for orig_img in original_images:
            # 转换为numpy数组并归一化到[-1,1]范围
            orig_img = torch.from_numpy(np.array(orig_img)).float()
            orig_img = orig_img / 127.5 - 1
            original_images_list.append(orig_img)
        original_images_list = torch.stack(original_images_list, dim=0)
        original_images_list = rearrange(original_images_list, "f h w c -> f c h w")
        original_images_list = original_images_list.to(device=device, dtype=latents.dtype) # 将original_images_list转换为与latents相同的设备和数据类型

        # 处理masks
        masks_list = []
        for mask in masks:
            # 转换为numpy数组，保持0-1范围
            mask = torch.from_numpy(np.array(mask)).float()
            mask = (mask > 0.5).float()
            # 扩展为3通道（如果需要的话）
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            masks_list.append(mask)
        masks_list = torch.stack(masks_list, dim=0)
        masks_list = rearrange(masks_list, "f h w c -> f c h w")
        masks_list = masks_list.to(device=device, dtype=latents.dtype) # 将masks_list转换为与latents相同的设备和数据类型
        # 先将原始图像转换为 latents
        original_image_latents = self.vae.encode(original_images_list).latent_dist.sample() * self.vae.config.scaling_factor
        # 调整masks大小以匹配latent空间
        masks_list = F.interpolate(masks_list, size=latents.shape[-2:], mode='bilinear')
        # 仅初始化 `mask=0` 的部分
        if masks is not None:
            latents = masks_list * original_image_latents + (1 - masks_list) * latents
        # if masks is not None:
        #     # 1. 使用inplace操作
        #     masks_list = F.interpolate(masks_list, size=latents.shape[-2:], mode='bilinear')
        #     masks_list.mul_(1.0)  # 确保在正确的设备上
            
        #     # 2. 分块处理，避免一次性占用太多显存
        #     chunk_size = 4  # 可以根据实际情况调整
        #     for i in range(0, latents.shape[1], chunk_size):
        #         end_idx = min(i + chunk_size, latents.shape[1])
        #         # 只处理当前块
        #         curr_masks = masks_list[:, i:end_idx]
        #         curr_orig = original_image_latents[:, i:end_idx]
        #         curr_latents = latents[:, i:end_idx]
                
        #         # 3. 使用inplace操作更新latents
        #         latents[:, i:end_idx] = curr_masks * curr_orig + (1 - curr_masks) * curr_latents
                
        #         # 4. 清理中间变量
        #         del curr_masks, curr_orig, curr_latents
        #         torch.cuda.empty_cache()

        # if masks is not None:
        #     # 1. 逐帧处理latents
        #     for frame_idx in range(latents.shape[1]):
        #         # 只处理单帧
        #         curr_mask = masks_list[:, frame_idx:frame_idx+1]
        #         curr_mask = F.interpolate(curr_mask, size=latents.shape[-2:], mode='bilinear')
        #         curr_mask = curr_mask.unsqueeze(2).repeat(1, 1, latents.shape[2], 1, 1)
        #         curr_orig = original_image_latents[:, frame_idx:frame_idx+1]
                
        #         # 直接在原始latents上修改，不创建额外的临时变量
        #         latents[:, frame_idx:frame_idx+1].mul_(1 - curr_mask)
        #         latents[:, frame_idx:frame_idx+1].add_(curr_orig * curr_mask)
        #         # 立即清理
        #         del curr_mask, curr_orig
        #         torch.cuda.empty_cache()
        

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0.0)

        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # 8. Denoising loop
        self._num_timesteps = len(timesteps)
        indices = [[0, *range(i + 1, min(i + tile_size, num_frames))] for i in
                   range(0, num_frames - tile_size + 1, tile_size - tile_overlap)]
        if indices[-1][-1] < num_frames - 1:
            indices.append([0, *range(num_frames - tile_size + 1, num_frames)])

        pose_pil_image_list = []
        for pose in image_pose:
            pose = torch.from_numpy(np.array(pose)).float()
            pose = pose / 127.5 - 1
            pose_pil_image_list.append(pose)
        pose_pil_image_list = torch.stack(pose_pil_image_list, dim=0)
        pose_pil_image_list = rearrange(pose_pil_image_list, "f h w c -> f c h w")


        # print(indices)  # [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
        # print(pose_pil_image_list.size())  # [16, 3, 512, 512]

        self.pose_net.to(device)
        self.unet.to(device)

        with torch.cuda.device(device):
            torch.cuda.empty_cache()

        with self.progress_bar(total=len(timesteps) * len(indices)) as progress_bar:
            for i, t in enumerate(timesteps):
                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # 加载模型时指定正确的dtype
                load_model_to_device(self.pose_net, device, latent_model_input.dtype)
                load_model_to_device(self.unet, device, latent_model_input.dtype)
                # Concatenate image_latents over channels dimension
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                # predict the noise residual
                noise_pred = torch.zeros_like(image_latents)
                noise_pred_cnt = image_latents.new_zeros((num_frames,))
                weight = (torch.arange(tile_size, device=device) + 0.5) * 2. / tile_size
                weight = torch.minimum(weight, 2 - weight)
                for idx in indices:
                    # classification-free inference
                    pose_latents = self.pose_net(pose_pil_image_list[idx].to(device=device, dtype=latent_model_input.dtype))
                    _noise_pred = self.unet(
                        latent_model_input[:1, idx],
                        t,
                        encoder_hidden_states=image_embeddings[:1],
                        added_time_ids=added_time_ids[:1],
                        pose_latents=None,
                        image_only_indicator=image_only_indicator,
                        return_dict=False,
                    )[0]
                    noise_pred[:1, idx] += _noise_pred * weight[:, None, None, None]

                    # normal inference
                    _noise_pred = self.unet(
                        latent_model_input[1:, idx],
                        t,
                        encoder_hidden_states=image_embeddings[1:],
                        added_time_ids=added_time_ids[1:],
                        pose_latents=pose_latents,
                        image_only_indicator=image_only_indicator,
                        return_dict=False,
                    )[0]
                    noise_pred[1:, idx] += _noise_pred * weight[:, None, None, None]

                    noise_pred_cnt[idx] += weight
                    progress_bar.update()
                noise_pred.div_(noise_pred_cnt[:, None, None, None])

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                new_latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                # **锁定 `mask=1` 的部分**
                if masks is not None:
                    masks_latent = masks_list.repeat(1, new_latents.shape[2]//masks_list.shape[1], 1, 1)
                    new_latents = masks_latent * original_image_latents + (1 - masks_latent) * new_latents
                # if masks is not None:
                #     # 计算需要重复的次数
                #     repeat_times = new_latents.shape[2]//masks_list.shape[1]
                    
                #     # 分块处理
                #     chunk_size = 4  # 可以根据实际情况调整
                #     for i in range(0, new_latents.shape[1], chunk_size):
                #         end_idx = min(i + chunk_size, new_latents.shape[1])
                        
                #         # 只处理当前块的数据
                #         curr_masks = masks_list[:, i:end_idx]
                #         curr_masks = curr_masks.repeat(1, repeat_times, 1, 1)
                #         curr_orig = original_image_latents[:, i:end_idx]
                #         curr_new_latents = new_latents[:, i:end_idx]
                        
                #         # 使用inplace操作更新new_latents
                #         new_latents[:, i:end_idx] = curr_masks * curr_orig + (1 - curr_masks) * curr_new_latents
                        
                #         # 清理中间变量
                #         del curr_masks, curr_orig, curr_new_latents
                #         torch.cuda.empty_cache()
                # if masks is not None:
                #     # 逐帧处理new_latents
                #     for frame_idx in range(new_latents.shape[1]):
                #         # 只处理单帧
                #         curr_mask = masks_list[:, frame_idx:frame_idx+1]
                #         curr_mask = F.interpolate(curr_mask, size=new_latents.shape[-2:], mode='bilinear')
                #         curr_mask = curr_mask.unsqueeze(2).repeat(1, 1, new_latents.shape[2], 1, 1)
                #         curr_orig = original_image_latents[:, frame_idx:frame_idx+1]
                        
                #         # 直接在原始new_latents上修改，使用inplace操作
                #         new_latents[:, frame_idx:frame_idx+1].mul_(1 - curr_mask)
                #         new_latents[:, frame_idx:frame_idx+1].add_(curr_orig * curr_mask)
                        
                #         # 立即清理中间变量
                #         del curr_mask, curr_orig
                #         torch.cuda.empty_cache()
                latents = new_latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                # self.pose_net.cpu()
                # self.unet.cpu()
                # self.face_encoder.cpu()
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()

        if not output_type == "latent":
            self.vae.decoder.to(device)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            # **mask=1 的部分直接保留原图**
            if masks is not None:
                # 确保维度匹配
                masks_pixel = F.interpolate(masks_list, size=frames.shape[-2:], mode='bilinear')
                masks_pixel = masks_pixel.unsqueeze(0)  # 添加 batch 维度
                masks_pixel = masks_pixel.permute(0, 2, 1, 3, 4)  # 调整维度顺序
                masks_pixel = masks_pixel.repeat(1, 3, 1, 1, 1)  # 重复到3个通道
                masks_pixel = masks_pixel.to(device)

                if original_images_list.ndim != frames.ndim:
                    original_images_list = original_images_list.unsqueeze(0)
                original_images_list = original_images_list.permute(0, 2, 1, 3, 4) 
                original_images_list = original_images_list.to(device) 
                frames = frames.to(device) 
                print("frames shape:", frames.shape)
                print("masks_pixel shape:", masks_pixel.shape)
                print("original_images_list shape:", original_images_list.shape)
                frames = masks_pixel * original_images_list + (1 - masks_pixel) * frames
            # if masks is not None:
            #     # 分块处理
            #     chunk_size = 4  # 可以根据实际情况调整
            #     for i in range(0, frames.shape[2], chunk_size):
            #         end_idx = min(i + chunk_size, frames.shape[2])
                    
            #         # 只处理当前块的masks和frames
            #         curr_masks = masks_list[:, i:end_idx]
            #         # 调整当前块masks的大小以匹配frames
            #         curr_masks_pixel = F.interpolate(curr_masks, size=frames.shape[-2:], mode='bilinear')
                    
            #         # 获取当前块的原始图像和生成的frames
            #         curr_orig = original_images_list[:, i:end_idx]
            #         curr_frames = frames[:, :, i:end_idx]
                    
            #         # 使用inplace操作更新frames
            #         frames[:, :, i:end_idx] = curr_masks_pixel * curr_orig + (1 - curr_masks_pixel) * curr_frames
                    
            #         # 清理中间变量
            #         del curr_masks, curr_masks_pixel, curr_orig, curr_frames
            #         torch.cuda.empty_cache()
            # 在decode_latents后的处理也使用逐帧方式
            # if masks is not None:
            #     # 逐帧处理frames
            #     for frame_idx in range(frames.shape[2]):
            #         curr_mask = masks_list[:, frame_idx:frame_idx+1]
            #         curr_mask = F.interpolate(curr_mask, size=frames.shape[-2:], mode='bilinear')
            #         curr_mask = curr_mask.unsqueeze(1)  # 添加channel维度
            #         curr_orig = original_images_list[:, frame_idx:frame_idx+1]
            #         curr_orig = curr_orig.unsqueeze(2)  # 添加time维度
                    
            #         # 直接在原始frames上修改
            #         frames[:, :, frame_idx:frame_idx+1].mul_(1 - curr_mask)
            #         frames[:, :, frame_idx:frame_idx+1].add_(curr_orig * curr_mask)
                    
            #         # 立即清理
            #         del curr_mask, curr_orig
            #         torch.cuda.empty_cache()
            # print(frames.size()) # [1, 3, 16, 512, 512]
            # print(latents.size()) # [1, 16, 4, 64, 64]
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
            # print(frames[0].size()) # [16, 3, 512, 512]
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return InferenceAnimationPipelineOutput(frames=frames)
