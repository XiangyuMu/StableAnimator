import math
import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.attention import Attention, FeedForward
from animation.modules.attention_processor_1 import HeadPoseCogVideoXAttnProcessor2_0

def reshape_tensor(x, heads):
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs, heads, length, -1)
    return x

class AattentionBlock(nn.Module):
    def __init__(self, dim=1024, attention_head_dim=64, num_attention_heads=16, attention_bias=True, attention_out_bias=True, qk_norm=True, dropout=0.0, ff_inner_dim=None, ff_bias=True, final_dropout=True, activation_fn="gelu-approximate"):
        super().__init__()
        self.norm_in_1 = nn.LayerNorm(dim)
        self.norm_in_2 = nn.LayerNorm(dim)

        self.attn = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=HeadPoseCogVideoXAttnProcessor2_0(),
        )
        self.norm_out_1 = nn.LayerNorm(dim)
        
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
    def forward(self, id_embeds, latents):
        latents = self.norm_in_1(latents)
        id_embeds = self.norm_in_2(id_embeds)
        latents_new = self.attn(hidden_states=latents, encoder_hidden_states=id_embeds)
        latents = latents + latents_new
        latents = self.norm_out_1(latents)
        # print('latents',latents.shape)
        latents = self.ff(latents)
        return latents



class FacePerceiver(torch.nn.Module):
    def __init__(
        self,
        depth=4,
        output_dim=1024,
    ):
        super().__init__()
        self.norm_out = torch.nn.LayerNorm(output_dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        AattentionBlock(dim=1024, attention_head_dim=64, num_attention_heads=16)
                    ]
                )
            )

    def forward(self, head_latent, pose_latent):
  
        for attn in self.layers:
            head_output = attn[0](head_latent, pose_latent)
            head_latent = head_output + head_latent
        return self.norm_out(head_latent)


class FusionFaceId(ModelMixin):
    def __init__(self, in_channels):
        super().__init__()
        
        self.conv_in = nn.Conv2d(
            in_channels,
            320,
            kernel_size=3,
            padding=1,
        )

        # self.pose_conv_in = nn.Conv2d(
        #     in_channels,
        #     320,
        #     kernel_size=3,
        #     padding=1,
        # )

        self.head_norm = torch.nn.GroupNorm(num_groups=32, num_channels=320, eps=1e-6)
        self.pose_norm = torch.nn.GroupNorm(num_groups=32, num_channels=320, eps=1e-6)
        self.head_pose_conv = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=320, out_channels=640, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),

            nn.Conv2d(in_channels=640, out_channels=640, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=640, out_channels=1280, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),

            nn.Conv2d(in_channels=1280, out_channels=1280, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=1280, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
        )

        self.pose_conv = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=320, out_channels=640, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),

            nn.Conv2d(in_channels=640, out_channels=640, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=640, out_channels=1280, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),

            nn.Conv2d(in_channels=1280, out_channels=1280, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=1280, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
        )

        self.final_proj = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1)
        self.pose_final_proj = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1)


        self.proj_in = nn.Linear(2048, 1024)
        self.pose_proj_in = nn.Linear(2048, 1024)

        self.proj = nn.Linear(512, 1024)

        self.pool = nn.AdaptiveAvgPool1d(3)
        self.pose_pool = nn.AdaptiveAvgPool1d(4)

        self.head_pose = FacePerceiver()
        # self.norm_1 = torch.nn.LayerNorm(1024)
        # self.norm_2 = torch.nn.LayerNorm(1024)
    

    def forward(self, headPose, head_latent, arcface_embedding, pose, shortcut=False, scale=1.0):
        head_latent = self.conv_in(head_latent)  # (b, 320, 64, 64)
        # print('head_latent',head_latent.shape)
        # print('headPose',headPose.shape)
        head_latent = head_latent+headPose
        head_latent = self.head_norm(head_latent)
        head_latent = self.head_pose_conv(head_latent)  # (b, 1024, 8, 8)
        head_latent = self.final_proj(head_latent)  # (b, 2048, 8, 8)
        inner_dim = head_latent.shape[1]

        batch_frames_head, _, height, width = head_latent.shape
        head_latent = head_latent.permute(0, 2, 3, 1).reshape(batch_frames_head, height * width, inner_dim)   # (b, 8*8, 2048)
        # print('head_latent',head_latent.shape)
        head_latent = self.proj_in(head_latent)     # (b, 8*8, 1024)
        head_latent = head_latent.permute(0, 2, 1)
        # print('head_latent',head_latent.shape)
        head_latent = self.pool(head_latent)
        # print(head_latent',head_latent.shape)    
        head_latent = head_latent.permute(0, 2, 1) # (b,3,1024)
        arcface_embedding = arcface_embedding.unsqueeze(0)
        arcface_embedding = self.proj(arcface_embedding)
        # print('arcface_embedding',arcface_embedding.shape)
        # print('head_latent',head_latent.shape)
        # print('arcface_embedding',arcface_embedding.shape)
        # print('head_latent',head_latent.shape)
        head_latent = torch.cat([head_latent , arcface_embedding],dim=1)  # (b, 4, 1024)

        batch_frames_pose, _, _, _ = pose.shape
        # print('pose',pose.shape)
        pose_latent = self.pose_norm(pose)
        pose_latent = self.pose_conv(pose_latent)  # (b, 1024, 8, 8)
        pose_latent = self.pose_final_proj(pose_latent)
        pose_latent = pose_latent.permute(0, 2, 3, 1).reshape(batch_frames_pose, height * width, inner_dim)
        pose_latent = self.pose_proj_in(pose_latent)
        pose_latent = pose_latent.permute(0, 2, 1)
        pose_latent = self.pose_pool(pose_latent)    
        pose_latent = pose_latent.permute(0, 2, 1) # (b,3,1024) 

        head_latent = head_latent.repeat_interleave(int(batch_frames_pose/batch_frames_head), dim=0)



        out = self.head_pose(head_latent, pose_latent)

        # print('out',out.shape)

        return out


def test_fusion_face_id():
    # Initialize the model
    model = FusionFaceId(in_channels=4)

    # Create dummy inputs
    headPose = torch.randn(1, 320, 64, 64)
    head_latent = torch.randn(1, 4, 64, 64)
    arcface_embedding = torch.randn(1, 512)
    pose = torch.randn(16, 320, 64, 64)

    # Forward pass
    output = model(headPose, head_latent, arcface_embedding, pose)

    # Check the output shape
    assert output.shape == (16, 4, 1024), f"Expected output shape (16, 3, 1024), but got {output.shape}"

if __name__ == "__main__":
    test_fusion_face_id()
    print("All tests passed.")
