# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07, device='cuda'):
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
        z = torch.cat((z_i, z_j), dim=0) # Shape: [2 * batch_size, embedding_dim]
        
        # Tính ma trận tương đồng cosine
        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        
        # Lấy các cặp tương đồng dương (positive pairs)
        sim_ij = torch.diag(sim_matrix, batch_size)
        sim_ji = torch.diag(sim_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0) # Shape: [2 * batch_size]
        
        # Tạo mask để loại bỏ các cặp dương khỏi mẫu số
        mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=torch.bool)).to(self.device)
        
        # Lấy các cặp âm (negative pairs)
        negatives = sim_matrix[mask].view(2 * batch_size, -1) # Shape: [2 * batch_size, 2 * batch_size - 2]
        
        # Ma trận logits
        logits = torch.cat((positives.unsqueeze(1), negatives), dim=1) # Shape: [2*B, 1+2B-2] = [2*B, 2B-1]
        logits /= self.temperature
        
        # Nhãn cho CrossEntropyLoss: cặp đầu tiên (positive) luôn là đúng
        labels = torch.zeros(2 * batch_size).to(self.device).long()
        
        loss = self.criterion(logits, labels)
        return loss / (2 * batch_size)
    

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, temperature=0.05):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, anchors, positives, negatives):
        """
        Tính triplet loss với cosine similarity
        
        Args:
            anchors: embedding tensor [B, D]
            positives: embedding tensor [B, D]
            negatives: embedding tensor [B, D]
        """
        anchors = F.normalize(anchors, p=2, dim=1)
        positives = F.normalize(positives, p=2, dim=1)
        negatives = F.normalize(negatives, p=2, dim=1)
        
        pos_sim = torch.sum(anchors * positives, dim=1) / self.temperature
        neg_sim = torch.sum(anchors * negatives, dim=1) / self.temperature
        
        losses = F.relu(neg_sim - pos_sim + self.margin)
        return losses.mean()