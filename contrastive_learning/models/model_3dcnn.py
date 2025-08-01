import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class ClassifierProjection(nn.Module):
    def __init__(
        self,
        num_classes=2,
        in_features=512,
        dropout_prob=0.25,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(x)


class CLProjection(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.CLprj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if hidden_dim >= 32 else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x: torch.Tensor):
        return self.CLprj(x)


class FightDetector3DCNN(nn.Module):
    def __init__(
        self,
        num_classes=2,
        hidden_size=512,
        dropout_prob=0.2,
        cl_projection_hidden_dim=512,
        cl_projection_out_dim=128,
        unfreeze_layer=10,
    ):
        super().__init__()

        weights = R2Plus1D_18_Weights.KINETICS400_V1
        self.backbone = r2plus1d_18(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.CLprj = CLProjection(
            in_features, cl_projection_hidden_dim, cl_projection_out_dim
        )

        self.classifier = ClassifierProjection(
            num_classes=2, in_features=in_features, dropout_prob=dropout_prob
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    # gỏ NTXentLoss
    # def forward_cl_training(self, x_augment_1: torch.Tensor, x_augment_2: torch.Tensor):
    #     # [batch_size, seq_len, C, H, W] -> [batch_size, C, seq_len, H, W]
    #     x_augment_1 = x_augment_1.permute(0, 2, 1, 3, 4)
    #     x_augment_2 = x_augment_2.permute(0, 2, 1, 3, 4)

    #     feature_1 = self.backbone(x_augment_1)
    #     feature_2 = self.backbone(x_augment_2)

    #     embedding_1 = self.CLprj(feature_1)
    #     embedding_2 = self.CLprj(feature_2)

    #     embedding_1 = F.normalize(embedding_1, p=2, dim=-1)
    #     embedding_2 = F.normalize(embedding_2, p=2, dim=-1)

    #     return embedding_1, embedding_2

    def forward_cl_training(self, x: torch.Tensor):
        # [batch_size, seq_len, C, H, W] -> [batch_size, C, seq_len, H, W]
        x = x.permute(0, 2, 1, 3, 4) 
        feature = self.backbone(x) 
        embedding = self.CLprj(feature) 
        embedding = F.normalize(embedding, p=2, dim=-1) 
        return embedding 

    def forward(self, x: torch.Tensor, mode: str = "supervised"):

        if mode == "contrastive":
            # x1, x2 = x
            # return self.forward_cl_training(x_augment_1=x1, x_augment_2=x2)
            return self.forward_cl_training(x=x)

        # x: [batch_size, seq_len, C, H, W] -> [batch_size, C, seq_len, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        features = self.backbone(x)  # -> [batch_size, in_features]
        logits = self.classifier(features)
        return logits

    def __freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def __unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def __unfreeze_top_layers(self, num_layers: int):
        # Unfreeze theo block thay vì toàn bộ
        layers_to_unfreeze = [
            self.backbone.layer3,
            self.backbone.layer4,
            self.backbone.avgpool
        ][-num_layers:]
        
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

    def prepare_for_finetuning_classifier(self, unfreeze_layers=3):
        self.__freeze_backbone()
        self.__unfreeze_top_layers(num_layers=unfreeze_layers)

    def prepare_for_finetuning_contrastive(self):
        self.__unfreeze_backbone()
