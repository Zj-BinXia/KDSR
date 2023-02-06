import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class KDLoss(nn.Module):
    """Knowledge distillation loss.
    Args:
        loss_weight (float): Loss weight for KD loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, temperature = 0.15):
        super(KDLoss, self).__init__()
    
        self.loss_weight = loss_weight
        self.temperature = temperature

    def forward(self, T_fea, S_fea):
        """
        Args:
            T_fea (List): contain shape (N, L) vector of BlindSR_TA. 
            S_fea (List): contain shape (N, L) vector of BlindSR_ST.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        loss_distill_dis = 0
        for i in range(len(T_fea)):
            student_distance = F.log_softmax(S_fea[i] / self.temperature, dim=1)
            teacher_distance = F.softmax(T_fea[i].detach()/ self.temperature, dim=1)
            loss_distill_dis += F.kl_div(
                        student_distance, teacher_distance, reduction='batchmean')
            #loss_distill_abs += nn.L1Loss()(S_fea[i], T_fea[i].detach())
        return self.loss_weight * loss_distill_dis

                