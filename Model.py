import torch
from torch import nn
from torchvision import models


class FamilyNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        model = models.resnet18(pretrained=True, progress=True)
        print(model)

        # Adapt last layer to multi-label task
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=model.fc.in_features, out_features=128),
            nn.Linear(in_features=128, out_features=n_classes)
        )

        self.base_model = model
        self.sigm = nn.Sigmoid()


    def forward(self, x):
        return self.sigm(self.base_model(x))