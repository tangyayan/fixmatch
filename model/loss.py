import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class FixMatchLoss(nn.Module):
    def __init__(self, config: Config):
        super(FixMatchLoss, self).__init__()
        self.lambda_u = config.lambda_u
        self.tao = config.tao
        self.T = config.T
    
    def forward(self, outputs_x, targets_x, outputs_u_weak, outputs_u_strong):
        # 计算有标签数据的交叉熵损失
        loss_x = F.cross_entropy(outputs_x, targets_x) # [B,]

        # 计算无标签数据的伪标签
        pseudo_labels = torch.softmax(outputs_u_weak.detach() / self.T, dim=1) # [B, num_classes]
        max_probs, pseudo_labels = torch.max(pseudo_labels, dim=1)
        mask = max_probs.ge(self.tao).float() # [B,]

        # debug
        valid_pseudo = pseudo_labels[mask.bool()]
        counts = torch.bincount(valid_pseudo, minlength=10) # [num_classes,

        # 计算无标签数据的损失
        loss_u = F.cross_entropy(outputs_u_strong, pseudo_labels, reduction="none") # [B,]
        loss_u = (loss_u * mask).mean()

        loss = loss_x + self.lambda_u * loss_u
        return loss, loss_x, loss_u, counts