import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


class TorchVisionRAFT(nn.Module):
    """
    封装 torchvision 的 raft_large 模型。
    输入必须是 [0, 1] 区间的 RGB 图像。
    """
    def __init__(self):
        super().__init__()
        weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=weights).eval().cuda()
        self.transforms = weights.transforms()

    @torch.no_grad()
    def forward(self, image1, image2):
        """
        image1, image2: (B, C, H, W), float32, in [0, 1]
        返回光流张量: (B, 2, H, W)
        """
        if image1.shape[1] == 1:
            image1 = image1.repeat(1, 3, 1, 1)
            image2 = image2.repeat(1, 3, 1, 1)

        input_pair = self.transforms(image1, image2)  # list of tensors
        flow_predictions = self.model(*input_pair)    # list of flows

        return flow_predictions[-1]  # 最终 refined flow


class OpticalFlowLoss(nn.Module):
    """
    计算预测视频和GT视频之间的光流差异损失。
    支持 l1, l2, smooth_l1。
    """
    def __init__(self, flow_model=None, loss_type='l1'):
        super().__init__()
        self.flow_model = flow_model if flow_model is not None else TorchVisionRAFT()
        self.loss_type = loss_type

    def compute_loss(self, flow1, flow2):
        if self.loss_type == 'l1':
            return F.l1_loss(flow1, flow2)
        elif self.loss_type == 'l2':
            return F.mse_loss(flow1, flow2)
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss(flow1, flow2)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def forward(self, pred_video, gt_video):
        """
        pred_video, gt_video: (B, T, C, H, W), float32, [0, 1]
        """
        B, T, C, H, W = pred_video.shape
        pred_flows, gt_flows = [], []

        for t in range(T - 1):
            pred_flow = self.flow_model(pred_video[:, t], pred_video[:, t + 1])  # (B, 2, H, W)
            gt_flow = self.flow_model(gt_video[:, t], gt_video[:, t + 1])        # (B, 2, H, W)

            pred_flows.append(pred_flow)
            gt_flows.append(gt_flow)

        pred_flows = torch.stack(pred_flows, dim=1)  # (B, T-1, 2, H, W)
        gt_flows = torch.stack(gt_flows, dim=1)

        return self.compute_loss(pred_flows, gt_flows)
