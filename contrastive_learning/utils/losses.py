# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07, device="cuda"):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        """
        z_i, z_j là hai batch embeddings đã được chuẩn hóa L2.
        Shape: [batch_size, embedding_dim]
        """
        batch_size = z_i.shape[0]

        # Kết hợp hai batch embeddings lại
        z = torch.cat((z_i, z_j), dim=0)  # Shape: [2 * batch_size, embedding_dim]

        # Tính ma trận tương đồng cosine
        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        # Lấy các cặp tương đồng dương (positive pairs)
        sim_ij = torch.diag(sim_matrix, batch_size)
        sim_ji = torch.diag(sim_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # Shape: [2 * batch_size]

        # Tạo mask để loại bỏ các cặp dương khỏi mẫu số
        mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=torch.bool)).to(
            self.device
        )

        # Lấy các cặp âm (negative pairs)
        negatives = sim_matrix[mask].view(
            2 * batch_size, -1
        )  # Shape: [2 * batch_size, 2 * batch_size - 2]

        # Ma trận logits
        logits = torch.cat(
            (positives.unsqueeze(1), negatives), dim=1
        )  # Shape: [2*B, 1+2B-2] = [2*B, 2B-1]
        logits /= self.temperature

        # Nhãn cho CrossEntropyLoss: cặp đầu tiên (positive) luôn là đúng
        labels = torch.zeros(2 * batch_size).to(self.device).long()

        loss = self.criterion(logits, labels)
        return loss / (2 * batch_size)


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, temperature=0.05, device="cuda"):
        super().__init__()
        self.margin = margin
        self.device = device
        self.temperature = temperature

    #     def forward(self, anchors, positives, negatives):
    #         """
    #         Tính triplet loss với cosine similarity

    #         Args:
    #             anchors: embedding tensor [B, D]
    #             positives: embedding tensor [B, D]
    #             negatives: embedding tensor [B, D]
    #         """

    #         # Chuyển tất cả inputs lên cùng device
    #         device = self.device
    #         positives = positives.to(device)
    #         negatives = negatives.to(device)

    #         # Chuyển margin và temperature thành tensors trên device
    #         margin_t = torch.tensor(self.margin, device=device)
    #         temp_t   = torch.tensor(self.temperature, device=device)

    #         # Chuẩn hoá embeddings
    #         anchors   = F.normalize(anchors,   p=2, dim=1)
    #         positives = F.normalize(positives, p=2, dim=1)
    #         negatives = F.normalize(negatives, p=2, dim=1)

    #         # Tính cosine similarities và chia temperature
    #         pos_sim = torch.sum(anchors * positives, dim=1) / temp_t
    #         neg_sim = torch.sum(anchors * negatives, dim=1) / temp_t

    #         # Triplet hinge loss
    #         losses = F.relu(neg_sim - pos_sim + margin_t)

    #         return losses.mean()

    def forward(self, anchor, positive, negative, margin=1.):
        # device = self.device
        # positives = positives.to(device)
        # negatives = negatives.to(device)
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        loss = F.relu(pos_dist - neg_dist + margin).mean()
        return loss
