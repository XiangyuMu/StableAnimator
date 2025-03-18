from pathlib import Path
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init     # 缺少这个导入
import numpy as np
from diffusers.models.modeling_utils import ModelMixin
from hiera import hiera_tiny_224
import types
import os

class ClothEncoder(ModelMixin):
    def __init__(self, noise_latent_channels=320):
        """构造基于Hiera的服装编码器
        Args:
            noise_latent_channels (int): 输出通道数,需要与UNet的block_out_channels[0]匹配
        """
        super().__init__()
        
        # # 指定预训练模型下载路径
        # cache_dir = os.path.expanduser("/data/xuzhenhao/StableAnimator/.cache/hiera")
        # os.makedirs(cache_dir, exist_ok=True)
        
        # 使用预训练的hiera_base_224模型，并指定预训练权重
        self.backbone = hiera_tiny_224(pretrained=True) # 指定使用哪个预训练checkpoint
        # self.backbone = torch.hub.load("facebookresearch/hiera", model="hiera_base_224", pretrained=True, checkpoint="mae_in1k_ft_in1k")
        # self.backbone = Hiera.from_pretrained("facebook/hiera_base_224.mae_in1k_ft_in1k")  # mae pt then in1k ft'd model
        
        
        # 调整通道数和特征图尺寸的层
        self.adapter = nn.Sequential(
            nn.Conv2d(768, noise_latent_channels, 1),  # 768 -> 320 channels
            nn.SiLU(),
            nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)  # 14x14 -> 64x64
        )
        
        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.ones(1) * 2)
        # 添加权重初始化
        self._initialize_weights()
    def _initialize_weights(self):
        """初始化adapter中的权重"""
        # 只初始化Conv2d层，因为Upsample没有可学习参数
        for m in self.adapter:
            if isinstance(m, nn.Conv2d):
                # He初始化
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                init.normal_(m.weight, mean=0.0, std=np.sqrt(2. / n))
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        """前向传播
        Args:
            x (torch.Tensor): 输入张量 [batch, frames, channels, 512, 512]
        Returns:
            torch.Tensor: 编码后的特征 [batch, frames, 320, 64, 64]
        """
        if x.ndim == 5:
            x = einops.rearrange(x, 'b f c h w -> (b f) c h w')
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 获取特征
        _, intermediates = self.backbone(x, return_intermediates=True)
        x = intermediates[-1]  # [B*F, 768, 14, 14]
        x = x.permute(0, 3, 1, 2)
        # 调整通道数和特征图尺寸
        x = self.adapter(x)  # [B*F, 320, 64, 64]
            
        return x * self.scale

    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        """加载预训练权重
        Args:
            pretrained_model_path (str, optional): 自定义预训练模型路径
        Returns:
            ClothEncoder: 加载了预训练权重的模型
        """
        if not Path(pretrained_model_path).exists():
            print(f"There is no model file in {pretrained_model_path}")
        print(f"loaded ClothEncoder's pretrained weights from {pretrained_model_path}.")
        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        model = ClothEncoder(noise_latent_channels=320)
        model.load_state_dict(state_dict, strict=True)
            
        return model