from pathlib import Path
import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.init as init
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.resnet import ResnetBlock2D, Downsample2D

class ClothEncoder(ModelMixin):
    def __init__(self, noise_latent_channels=320):
        """构造服装编码器
        Args:
            noise_latent_channels (int): 输出通道数,需要与UNet的block_out_channels[0]匹配
        """
        super().__init__()
        
        # 定义通道数配置
        self.channels = [32, 64, 128, 256]
        in_channels = 3  # RGB输入
        
        # 初始卷积层
        self.conv_in = nn.Conv2d(in_channels=in_channels, out_channels=self.channels[0], kernel_size=3, padding=1)
        
        # ResNet块和下采样层
        self.resnets = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        for i in range(len(self.channels)):
            in_ch = self.channels[i-1] if i > 0 else self.channels[0]
            out_ch = self.channels[i]
            # # 根据通道数动态调整groups
            # groups = min(32, in_ch, out_ch)
            # groups = groups if in_ch % groups == 0 and out_ch % groups == 0 else 1
            # 添加ResNet块
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    temb_channels=None,  # 不使用时间嵌入
                )
            )
            
            # 添加下采样层(除了最后一层)
            if i != len(self.channels) - 1:
                self.downsamplers.append(
                    Downsample2D(
                        channels=out_ch,
                        use_conv=True,
                        out_channels=out_ch,
                        padding=1,
                        name="op"
                    )
                )
            else:
                self.downsamplers.append(nn.Identity())

        # 最终投影层
        self.final_proj = nn.Conv2d(self.channels[-1], noise_latent_channels, kernel_size=1)
        
        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.ones(1) * 2)
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        # 初始化conv_in
        init.normal_(self.conv_in.weight, mean=0.0, std=np.sqrt(2. / (9 * self.channels[0])))
        if self.conv_in.bias is not None:
            init.zeros_(self.conv_in.bias)
            
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
            
        x = self.conv_in(x)
        
        # 通过ResNet块和下采样层
        for resnet, downsampler in zip(self.resnets, self.downsamplers):
            x = resnet(x, temb=None)
            x = downsampler(x)
            
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