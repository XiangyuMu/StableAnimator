from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from diffusers.models.modeling_utils import ModelMixin

class ClothEncoder(ModelMixin):
    def __init__(self, noise_latent_channels=320):
        """构造服装编码器
        Args:
            noise_latent_channels (int): 输出通道数,需要与UNet的block_out_channels[0]匹配
        """
        super().__init__()
        # 卷积层序列
        self.conv_layers = nn.Sequential(
            # 第一个编码块: 3->16 channels, 1/2分辨率
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.BatchNorm2d(16),

            # 第二个编码块: 16->32 channels, 1/4分辨率
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1), 
            nn.SiLU(),
            nn.BatchNorm2d(32),

            # 第三个编码块: 32->64 channels, 1/8分辨率
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.BatchNorm2d(64),

            # 第四个编码块: 64->128 channels, 保持分辨率
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm2d(128)
        )

        # 最终投影层,将特征投影到目标维度
        self.final_proj = nn.Conv2d(in_channels=128, out_channels=noise_latent_channels, kernel_size=1)

        # 初始化权重
        self._initialize_weights()

        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.ones(1) * 2)

    def _initialize_weights(self):
        """使用He初始化权重,偏置初始化为0"""
        for m in self.conv_layers:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                init.normal_(m.weight, mean=0.0, std=np.sqrt(2. / n))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)
                
        # 最终投影层初始化为0
        init.zeros_(self.final_proj.weight)
        if self.final_proj.bias is not None:
            init.zeros_(self.final_proj.bias)

    def forward(self, x):
        """前向传播
        Args:
            x (torch.Tensor): 输入张量
                如果是5D输入 [batch, frames, channels, height, width]
                如果是4D输入 [batch, channels, height, width]
        Returns:
            torch.Tensor: 编码后的特征
        """
        if x.ndim == 5:
            x = einops.rearrange(x, "b f c h w -> (b f) c h w")
        x = self.conv_layers(x)
        x = self.final_proj(x)
        return x * self.scale

    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        """加载预训练权重
        Args:
            pretrained_model_path (str): 预训练模型路径
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