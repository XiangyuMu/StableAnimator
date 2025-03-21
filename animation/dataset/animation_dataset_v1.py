import os, io, csv, math, random
from importlib.metadata import files
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from einops import rearrange
import cv2
import warnings


class LargeScaleAnimationVideos(Dataset):
    def __init__(self, root_path, txt_path, width, height, n_sample_frames, sample_frame_rate, sample_margin=30,
                 app=None, handler_ante=None, face_helper=None):
        self.root_path = root_path
        self.txt_path = txt_path
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_frame_rate = sample_frame_rate
        self.sample_margin = sample_margin

        self.video_files = self._read_txt_file_images()

        self.app = app
        self.handler_ante = handler_ante
        self.face_helper = face_helper

    def _read_txt_file_images(self):
        with open(self.txt_path, 'r') as file:
            lines = file.readlines()
            video_files = []
            for line in lines:
                video_file = line.strip()
                video_files.append(video_file)
        return video_files

    def __len__(self):
        return len(self.video_files)

    def frame_count(self, frames_path):
        files = os.listdir(frames_path)
        png_files = [file for file in files if file.endswith('.png') or file.endswith('.jpg')]
        png_files_count = len(png_files)
        return png_files_count

    def find_frames_list(self, frames_path):
        files = os.listdir(frames_path)
        image_files = [file for file in files if file.endswith('.png') or file.endswith('.jpg')]
        if image_files[0].startswith('frame_'):
            image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        else:
            image_files.sort(key=lambda x: int(x.split('.')[0]))
        return image_files

    def get_face_masks(self, pil_img):
        rgb_image = np.array(pil_img)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        image_info = self.app.get(bgr_image)
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        if len(image_info) > 0:
            for info in image_info:
                x_1 = info['bbox'][0]
                y_1 = info['bbox'][1]
                x_2 = info['bbox'][2]
                y_2 = info['bbox'][3]
                cv2.rectangle(mask, (int(x_1), int(y_1)), (int(x_2), int(y_2)), (255), thickness=cv2.FILLED)
            mask = mask.astype(np.float64) / 255.0
        else:
            self.face_helper.clean_all()
            with torch.no_grad():
                bboxes = self.face_helper.face_det.detect_faces(bgr_image, 0.97)
            if len(bboxes) > 0:
                for bbox in bboxes:
                    cv2.rectangle(mask, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255),
                                  thickness=cv2.FILLED)
                mask = mask.astype(np.float64) / 255.0
            else:
                mask = np.ones((self.height, self.width), dtype=np.uint8)
        return mask

    def __getitem__(self, idx):

        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)

        frames_path = osp.join(self.video_files[idx], "images")
        heads_path = osp.join(self.video_files[idx], "centered_heads")
        poses_wo_head_path = osp.join(self.video_files[idx], "pose_wo_head")
        clothes_path = osp.join(self.video_files[idx], "clothes_white_complete")
        face_masks_path = osp.join(self.video_files[idx], "faces")
        poses_with_head_path = osp.join(self.video_files[idx], "pose_head_new")
        video_length = self.frame_count(frames_path)
        frames_list = self.find_frames_list(frames_path)

        clip_length = min(video_length, (self.n_sample_frames - 1) * self.sample_frame_rate + 1)

        # print("-------------------------------")
        # print(clip_length)
        # print(video_length)
        # print(type(random))
        # print("-------------------------------")

        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        ).tolist()
        all_indices = list(range(0, video_length))
        available_indices = [i for i in all_indices if i not in batch_index]
        reference_frame_idx = None
        if available_indices:
            reference_frame_idx = random.choice(available_indices)
        else:
            print("There is no available frame")
            extreme_sample_frame_rate = 2
            extreme_clip_length = min(video_length, (self.n_sample_frames - 1) * extreme_sample_frame_rate + 1)
            extreme_start_idx = random.randint(0, video_length - extreme_clip_length)
            extreme_batch_index = np.linspace(
                extreme_start_idx, extreme_start_idx + extreme_clip_length - 1, self.n_sample_frames, dtype=int
            ).tolist()
            extreme_available_indices = [i for i in all_indices if i not in extreme_batch_index]
            if extreme_available_indices:
                reference_frame_idx = random.choice(extreme_available_indices)
            else:
                print("There is no available frame in the extreme circumstance")
                print(frames_path)
                print(1 / 0)

        pose_wo_head_pil_image_list = []
        clothes_pil_image_list = []
        tgt_pil_image_list = []       # target image
        tgt_face_masks_list = []

        reference_frame_path = osp.join(heads_path, frames_list[reference_frame_idx])
        reference_pil_image = Image.open(reference_frame_path).convert('RGB')
        reference_pil_image = reference_pil_image.resize((self.width, self.height))
        reference_pil_image = torch.from_numpy(np.array(reference_pil_image)).float()
        reference_pil_image = reference_pil_image / 127.5 - 1      # normalize to [-1, 1]   reference image
        
        reference_cloth_path = osp.join(clothes_path, frames_list[reference_frame_idx])
        reference_cloth_pil_image = Image.open(reference_cloth_path).convert('RGB')
        reference_cloth_pil_image = reference_cloth_pil_image.resize((self.width, self.height))
        reference_cloth_pil_image = torch.from_numpy(np.array(reference_cloth_pil_image)).float()
        reference_cloth_pil_image = reference_cloth_pil_image / 127.5 - 1      # normalize to [-1, 1]   clothes image

        reference_head_path = osp.join(heads_path, frames_list[reference_frame_idx])
        reference_head_pil_image = Image.open(reference_head_path).convert('RGB')
        reference_head_pil_image = reference_head_pil_image.resize((self.width, self.height))
        reference_head_pil_image = torch.from_numpy(np.array(reference_head_pil_image)).float()
        reference_head_pil_image = reference_head_pil_image / 127.5 - 1      # normalize to [-1, 1]   head image


        reference_poses_with_head_path = osp.join(poses_with_head_path, frames_list[reference_frame_idx])
        reference_poses_with_head_pil_image = Image.open(reference_poses_with_head_path).convert('RGB')
        reference_poses_with_head_pil_image = reference_poses_with_head_pil_image.resize((self.width, self.height))
        reference_poses_with_head_pil_image = torch.from_numpy(np.array(reference_poses_with_head_pil_image)).float()
        reference_poses_with_head_pil_image = reference_poses_with_head_pil_image / 127.5 - 1      # normalize to [-1, 1]   pose with head image


        self.face_helper.clean_all()
        # print('reference_frame_path',reference_frame_path)
        reference_frame_face = cv2.imread(reference_frame_path)
        reference_frame_face = cv2.resize(reference_frame_face, (self.width, self.height))
        # reference_frame_face_bgr = cv2.cvtColor(reference_frame_face, cv2.COLOR_RGB2BGR)
        reference_frame_face_info = self.app.get(reference_frame_face)
        if len(reference_frame_face_info) > 0:
            # print('detect face using insightface')
            reference_frame_face_info = sorted(reference_frame_face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
            reference_frame_id_ante_embedding = reference_frame_face_info['embedding']
        else:
            reference_frame_id_ante_embedding = None

        if reference_frame_id_ante_embedding is None:
            self.face_helper.read_image(reference_frame_face)
            self.face_helper.get_face_landmarks_5(only_center_face=True)
            self.face_helper.align_warp_face()

            if len(self.face_helper.cropped_faces) == 0:
                reference_frame_id_ante_embedding = np.zeros((512,))
            else:
                reference_frame_align_face = self.face_helper.cropped_faces[0]
                print('fail to detect face using insightface, extract embedding on align face')
                # reference_frame_id_ante_embedding = self.handler_ante.get_feat(reference_frame_align_face)
                reference_frame_id_ante_embedding = np.zeros((512,))

        for index in batch_index:
            tgt_img_path = osp.join(frames_path, frames_list[index])
            pose_name = os.path.splitext(os.path.basename(tgt_img_path))[0]
            pose_name = pose_name + '.png'
            face_name = pose_name
            cloth_name = pose_name
            pose_path = osp.join(poses_wo_head_path, pose_name)
            cloth_path = osp.join(clothes_path, cloth_name)
            face_mask_path = osp.join(face_masks_path, face_name)

            try:
                tgt_img_pil = Image.open(tgt_img_path).convert('RGB')
            except Exception as e:
                print(f"Fail loading the image: {tgt_img_path}")

            # tgt_face_mask = self.get_face_masks(tgt_img_pil)

            try:
                tgt_face_mask = Image.open(face_mask_path)
                tgt_face_mask = tgt_face_mask.resize((self.width, self.height))
                tgt_face_mask = torch.from_numpy(np.array(tgt_face_mask)).float()
                tgt_face_mask = tgt_face_mask / 255     # normalize to [0, 1]
            except Exception as e:
                print(f"Fail loading the face masks: {face_mask_path}")
                tgt_face_mask = torch.ones(self.height, self.width, 1)
            tgt_face_masks_list.append(tgt_face_mask)

            tgt_img_pil = tgt_img_pil.resize((self.width, self.height))
            tgt_img_tensor = torch.from_numpy(np.array(tgt_img_pil)).float()
            tgt_img_normalized = tgt_img_tensor / 127.5 - 1      # normalize to [-1, 1]
            tgt_pil_image_list.append(tgt_img_normalized)

            try:
                pose = Image.open(pose_path).convert('RGB')
                pose = pose.resize((self.width, self.height))
                pose = torch.from_numpy(np.array(pose)).float()
                pose = pose / 127.5 - 1     # normalize to [-1, 1]
            except Exception as e:
                print(f"Fail loading the poses: {pose_path}")
                pose = torch.zeros_like(reference_pil_image)
            pose_wo_head_pil_image_list.append(pose)

            try:
                clothes = Image.open(cloth_path).convert('RGB')
                clothes = clothes.resize((self.width, self.height))
                clothes = torch.from_numpy(np.array(clothes)).float()
                clothes = clothes / 127.5 - 1     # normalize to [-1, 1]
            except Exception as e:
                print(f"Fail loading the poses: {cloth_path}")
                clothes = torch.zeros_like(reference_pil_image)
            clothes_pil_image_list.append(clothes)   

        # pose_pil_image_list = torch.stack(pose_pil_image_list, dim=0)
        clothes_pil_image_list = torch.stack(clothes_pil_image_list, dim=0)
        tgt_pil_image_list = torch.stack(tgt_pil_image_list, dim=0)
        pose_wo_head_pil_image_list = torch.stack(pose_wo_head_pil_image_list, dim=0)
        tgt_pil_image_list = rearrange(tgt_pil_image_list, "f h w c -> f c h w")
        # reference_pil_image = rearrange(reference_pil_image, "h w c -> c h w")
        reference_cloth_pil_image = rearrange(reference_cloth_pil_image, "h w c -> c h w")
        reference_head_pil_image = rearrange(reference_head_pil_image, "h w c -> c h w")
        reference_poses_with_head_pil_image = rearrange(reference_poses_with_head_pil_image, "h w c -> c h w")
        # pose_pil_image_list = rearrange(pose_pil_image_list, "f h w c -> f c h w")
        clothes_pil_image_list = rearrange(clothes_pil_image_list, "f h w c -> f c h w")
        pose_wo_head_pil_image_list = rearrange(pose_wo_head_pil_image_list, "f h w c -> f c h w")

        tgt_face_masks_list = torch.stack(tgt_face_masks_list, dim=0)
        tgt_face_masks_list = torch.unsqueeze(tgt_face_masks_list, dim=-1)
        tgt_face_masks_list = rearrange(tgt_face_masks_list, "f h w c -> f c h w")

        sample = dict(
            pixel_values=tgt_pil_image_list,     # target image list
            # reference_image=reference_pil_image,    # head image
            reference_cloth_image=reference_cloth_pil_image,    # clothes image
            pose_pixels=pose_wo_head_pil_image_list,     # pose image list
            clothes_pixels=clothes_pil_image_list,    # clothes image list
            faceid_embeds=reference_frame_id_ante_embedding,    # face embedding
            tgt_face_masks=tgt_face_masks_list,    # face mask list
            reference_head_image=reference_head_pil_image,    # head image
            pose_with_head_image=reference_poses_with_head_pil_image,    # pose with head image
        )
        return sample

import sys
sys.path.append(osp.join(osp.dirname(__file__), '..', '..'))
from animation.modules.face_model import FaceModel

if __name__ == "__main__":
    face_model = FaceModel()
    dataset = LargeScaleAnimationVideos(
        root_path="animation_data",
        txt_path="animation_data/video_path.txt",
        width=512,
        height=512,
        n_sample_frames=16,
        sample_frame_rate=4,
        sample_margin=30,
        app=face_model.app,
        handler_ante=face_model.handler_ante,
        face_helper=face_model.face_helper
    )

    print(f"Dataset length: {len(dataset)}")
    for i in range(len(dataset)):
        print(f"Testing sample {i}")
        try:
            sample = dataset[i]
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: {value.shape}")
                else:
                    print(f"{key}: {value.shape if hasattr(value, 'shape') else len(value)}")
            break
        except Exception as e:
            print(f"Error processing sample {i}: {e}")



