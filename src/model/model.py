import torch
import torch.nn as nn
from torchviz import make_dot
from configs.configs import parse_arguments
from torchvision import models


args = parse_arguments()


class FightDetectionModel(nn.Module):
    def __init__(
        self, num_classes=2, hidden_size=args.hidden_size, dropout_prob=args.dropout_prob
    ):

        self.mobilenet = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )
        self.feature_extractor = nn.Sequential(*list(self.mobilenet.children())[:-1])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.dropout1 = nn.Dropout(dropout_prob)

        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout2 = nn.Dropout(dropout_prob)

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(32, num_classes),
        )

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)
        with torch.no_grad():
            features = self.feature_extractor(x)
        features.mean([2, 3])
        features.view(batch_size, seq_len, -1)
        features = self.dropout1(features)

        lstm_out, _ = self.lstm(features)
        last_time_step = lstm_out[:-1:]
        out = self.dropout2(last_time_step)
        out = self.fc_layers(out)
        # out = self.softmax(out)

        return out


    def plot_model_structure(self, input_shape=(1, 10, 3, 224, 224), save_path='./visualization/FightDetectionModel.png'):

        dummy_input = torch.randn(*input_shape)

        self.eval()
        with torch.no_grad():
            output = self.forward(dummy_input)
        self.train() 

        dot = make_dot(output, params=dict(self.named_parameters()))

        dot.render(save_path.replace('.png', ''), format='png', cleanup=True)
        print(f"Biểu đồ mô hình đã được lưu tại {save_path}")

