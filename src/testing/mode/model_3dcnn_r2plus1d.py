import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class FightDetector3DCNN(nn.Module):
    def __init__(
        self,
        num_classes=2,
        hidden_size=512,
        dropout_prob=0.25,
        image_height=64,
        image_width=64,
        unfreeze_layer=10,
    ):
        super().__init__()

        weights = R2Plus1D_18_Weights.KINETICS400_V1
        self.backbone = r2plus1d_18(weights=weights)

        for param in self.backbone.parameters():
            param.requires_grad = False

        total_layer = len(list(self.backbone.children()))
        layers_to_unfreeze = list(self.backbone.children())[-unfreeze_layer:]

        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor):
        # x: [batch_size, seq_len, C, H, W] -> [batch_size, C, seq_len, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        features = self.backbone(x)  # -> [batch_size, in_features]
        logits = self.classifier(features)
        return logits
