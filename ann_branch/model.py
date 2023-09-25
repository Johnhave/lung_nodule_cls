# --coding:utf-8--
import torch.nn as nn




class ANN(nn.Module):
    def __init__(self, num_classes):
        super(ANN, self).__init__()
        self.fc = nn.Sequential(nn.Linear(33, 1024),
                                nn.ReLU(inplace=True),
                                nn.Linear(1024, num_classes))  # the last layer
    def forward(self, x):
        out = self.fc(x)
        return out



