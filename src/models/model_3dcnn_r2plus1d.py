import torch.nn as nn
from configs.configs import parse_arguments
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

args = parse_arguments()


class FightDetection3DCNN(nn.Module):
    def __init__(
        self,
        num_classes=args.num_classes,
        hidden_size=args.hidden_size,
        dropout_prob=args.dropout_prob,
        image_height=args.image_height,
        image_width=args.image_width,
        unfreeze_layers=10,  # Số layer cuối sẽ unfreeze
    ):
        super().__init__()

        # 1) Backbone R(2+1)D với pretrained weights
        weights = R2Plus1D_18_Weights.KINETICS400_V1
        self.backbone = r2plus1d_18(weights=weights)

        # 2) Đóng băng toàn bộ backbone trước
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 3) Mở đóng băng cho các layer cuối
        total_layers = len(list(self.backbone.children()))
        layers_to_unfreeze = list(self.backbone.children())[-unfreeze_layers:]

        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

        # 4) Thay thế lớp FC cuối bằng classifier mới
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Loại bỏ FC gốc

        # 5) Bộ phân lớp với regularization mạnh
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.LayerNorm(256),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, C, H, W) -> (batch, C, seq_len, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        # Extract features
        features = self.backbone(x)  # (batch, in_features)

        # Classification
        logits = self.classifier(features)
        return logits
