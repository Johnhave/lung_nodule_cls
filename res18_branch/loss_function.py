# --coding:utf-8--
import torch.nn as nn
import torch
import torch.nn.functional as F


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class WeightedABSLoss(nn.Module):
    " weighted version of abs Loss"
    def __init__(self, device, alpha=10):
        super(WeightedABSLoss, self).__init__()
        self.alpha = alpha
        self.device = device

    def forward(self, inputs, targets):
        weight = torch.ones(inputs.shape).to(self.device)
        idx = targets == 1
        weight[idx] = self.alpha
        loss_all = torch.abs(inputs - targets) * weight
        return torch.mean(loss_all)

if __name__ == '__main__':
    a = torch.tensor([[1], [0], [0], [0]])
    b = torch.tensor([[1], [0], [1], [0]])
    lossfunc = WeightedABSLoss(alpha=10)
    loss = lossfunc(a, b)
    print(loss)