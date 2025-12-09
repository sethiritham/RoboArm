import torch
import torch.nn as nn
from torchvision import models

class RoboticArmNet(nn.Module):
    def __init__(self, pretrained=True):
        super(RoboticArmNet, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)
        
        # Output: X,Y,Z,Grip
        self.model.fc = nn.Linear(self.model.fc.in_features, 4) 

    def forward(self, x):
        return self.model(x)

def get_model(device):
    model = RoboticArmNet()
    model = model.to(device)
    return model