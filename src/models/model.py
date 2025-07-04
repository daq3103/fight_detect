import torch
import torch.nn as nn
from torchviz import make_dot
from configs.configs import parse_arguments
from torchvision import models
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


args = parse_arguments()


class FightDetectionModel(nn.Module):
    def __init__(
        self,
        num_classes=args.num_classes,
        hidden_size=args.hidden_size,
        dropout_prob=args.dropout_prob,
        image_height=args.image_height,
        image_width=args.image_width,
    ):
        super().__init__()

        # 1) Backbone MobileNetV2
        backbone = models.mobilenet_v2(pretrained=True).features
        for param in backbone.parameters():
            param.requires_grad = False
        # Unfreeze 40 leaf modules
        leafs = [m for m in backbone.modules() if len(list(m.children())) == 0]
        for module in leafs[-40:]:
            for p in module.parameters():
                p.requires_grad = True
        self.backbone = backbone

        # 2) Tính kích thước đầu ra của feature extractor
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_height, image_width)
            fo = self.backbone(dummy)
            feat_dim = fo.view(1, -1).size(1)

        # 3) LSTM Bi-directional
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        # 4) Cụm classifier gọn trong Sequential (bỏ Softmax)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(2 * hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, C, H, W)
        B, S, C, H, W = x.shape
        # 1) Backbone per-frame
        x = x.view(B * S, C, H, W)
        x = self.backbone(x)  # (B*S, feat_map_dims...)
        x = x.view(B, S, -1)  # (B, S, feat_dim)

        # 2) LSTM
        lstm_out, _ = self.lstm(x)  # (B, S, 2*hidden_size)
        fw = lstm_out[:, -1, : self.lstm.hidden_size]
        bw = lstm_out[:, 0, self.lstm.hidden_size :]
        x = torch.cat([fw, bw], dim=1)  # (B, 2*hidden_size)

        logits = self.classifier(x)
        return logits

    # def plot_model_structure(
    #     self,
    #     input_shape=(1, 10, 3, 224, 224),
    #     save_path="./visualization/FightDetectionModel.png",
    # ):

    #     dummy_input = torch.randn(*input_shape)

    #     self.eval()
    #     with torch.no_grad():
    #         output = self.forward(dummy_input)
    #     self.train()

    #     dot = make_dot(output, params=dict(self.named_parameters()))

    #     dot.render(save_path.replace(".png", ""), format="png", cleanup=True)
    #     print(f"Biểu đồ mô hình đã được lưu tại {save_path}")


class FightDetection3DCNN(nn.Module):
    def __init__(
        self,
        num_classes=args.num_classes,
        dropout_prob=args.dropout_prob,
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

    def forward(self, x):
        # x: (batch, seq_len, C, H, W) -> (batch, C, seq_len, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        # Extract features
        features = self.backbone(x)  # (batch, in_features)

        # Classification
        logits = self.classifier(features)
        return logits
