import torch
import torch.nn as nn
import torchvision.models as models

class BananaOrAppleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)

        # Freeze all first
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze last residual block
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.base_model(x)