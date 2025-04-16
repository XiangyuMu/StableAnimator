import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGGLoss(nn.Module):
    """
    VGG损失函数类，使用预训练的VGG19网络提取特征并计算感知损失。
    """
    def __init__(self, device, vgg_model='vgg19', weights=None, layers=None, normalize=True):
        super(VGGLoss, self).__init__()
        
        if vgg_model == 'vgg19':
            self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        elif vgg_model == 'vgg16':
            self.vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        else:
            raise ValueError(f"不支持的VGG模型: {vgg_model}")
        
        self.vgg = self.vgg.to(device).eval()
        
        # 冻结VGG参数
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # 默认使用的特征层
        self.layers = layers or {'3': 1.0, '8': 1.0, '17': 1.0, '26': 1.0, '35': 1.0}
        
        self.normalize = normalize
        
        # 图像标准化参数 (ImageNet均值和标准差)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.criterion = nn.MSELoss()
    
    def forward(self, x, y):
        """
        计算输入图像x和目标图像y之间的VGG感知损失
        
        参数:
            x: 输入图像, 形状为[B, C, H, W]
            y: 目标图像, 形状为[B, C, H, W]
            
        返回:
            总的加权感知损失
        """
        if x.shape != y.shape:
            raise ValueError(f"输入和目标形状不匹配: {x.shape} vs {y.shape}")
        
        # 将像素值从[-1,1]转换到[0,1]
        if x.min() < 0:
            x = (x + 1) / 2
        if y.min() < 0:
            y = (y + 1) / 2
        
        print(f"x device: {x.device}")
        print(f"y device: {y.device}")

        mean = self.mean.to(x.device)
        std = self.std.to(x.device)

        print(f"mean device: {mean.device}")
        print(f"std device: {std.device}")
        
        # 应用ImageNet标准化
        if self.normalize:
            x = (x - mean) / std
            y = (y - mean) / std
        
        # 提取各层特征并计算损失
        x_features = self._get_features(x)
        y_features = self._get_features(y)
        
        loss = 0.0
        for layer_id, weight in self.layers.items():
            # 计算每一层的MSE损失并加权求和
            loss += weight * self.criterion(x_features[layer_id], y_features[layer_id])
            
        return loss
    
    def _get_features(self, x):
        """
        从输入图像中提取VGG特征
        """
        features = {}
        current = x
        
        # 遍历VGG网络的所有层
        for name, layer in self.vgg._modules.items():
            current = layer(current)
            if name in self.layers:
                features[name] = current
                
        return features

# 创建一个简单的接口以支持多帧的VGG损失计算
class VGGLossForVideos(nn.Module):
    def __init__(self, device):
        super(VGGLossForVideos, self).__init__()
        self.vgg_loss = VGGLoss(device)
        
    def forward(self, x, y):
        """
        计算视频帧之间的VGG损失
        
        参数:
            x: 输入视频张量, 形状为[B, F, C, H, W]或[B, C, F, H, W]
            y: 目标视频张量, 形状同x
            
        返回:
            所有帧的平均VGG损失
        """
        # 如果输入是[B, F, C, H, W]格式，需要重排为[B*F, C, H, W]
        if len(x.shape) == 5 and x.shape[1] > x.shape[2]:  # 假设第二维大于第三维表示[B, F, C, H, W]
            b, f, c, h, w = x.shape
            x_reshaped = x.reshape(b*f, c, h, w)
            y_reshaped = y.reshape(b*f, c, h, w)

        # 如果输入是[B, C, F, H, W]格式，需要转置和重排
        elif len(x.shape) == 5:
            b, c, f, h, w = x.shape
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(b*f, c, h, w)
            y_reshaped = y.permute(0, 2, 1, 3, 4).reshape(b*f, c, h, w)
        else:
            x_reshaped = x
            y_reshaped = y
        print(f"x_reshaped device: {x_reshaped.device}")
        print(f"y_reshaped device: {y_reshaped.device}")
            
        return self.vgg_loss(x_reshaped, y_reshaped)


# 创建__init__.py文件，使其成为一个包
def create_init_file():
    import os
    init_path = os.path.join(os.path.dirname(__file__), '__init__.py')
    with open(init_path, 'w') as f:
        f.write('# 这是animation.loss包的初始化文件\n')
        f.write('from .vgg_loss import VGGLoss, VGGLossForVideos\n')

# 当作为模块导入时不执行此操作
if __name__ == "__main__":
    create_init_file()