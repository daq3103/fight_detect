import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EfficientSTGCNBlock(nn.Module):
    """Khối ST-GCN hiệu quả với depthwise separable conv và ma trận học được"""
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()
        t_kernel_size = 9
        t_padding = (t_kernel_size - 1) // 2
        
        # Graph convolution (1x1 conv)
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Temporal convolution (depthwise separable)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(t_kernel_size, 1),
                stride=(stride, 1),
                padding=(t_padding, 0),
                groups=out_channels  # Depthwise
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)  # Pointwise
        )
        
        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
            nn.BatchNorm2d(out_channels)
        ) if residual and (in_channels != out_channels or stride != 1) else None
        
        # Ma trận kề có thể học (với khởi tạo từ A)
        self.register_buffer('A_base', A)  # Ma trận cơ sở không học được
        self.A_learned = nn.Parameter(A.clone())  # Ma trận học được
        self.attention = nn.Parameter(torch.ones(A.size(0), dtype=torch.float32))  # Học trọng số theo đỉnh

    def forward(self, x):
        res = self.residual(x) if self.residual is not None else 0
        
        # Tạo ma trận kề động: kết hợp base + learned + attention
        A = self.A_base * self.A_learned * self.attention[None, :]
        
        # Chuẩn hóa ma trận kề
        D = torch.sum(A, dim=1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0
        A_norm = D_inv_sqrt.diag() @ A @ D_inv_sqrt.diag()
        
        # Graph convolution
        x = torch.einsum('nctv,vw->nctw', (x, A_norm))
        x = self.gcn(x)
        
        # Temporal convolution
        x = self.tcn(x) + res
        return x

class OptimizedFightDetector(nn.Module):
    """Phiên bản tối ưu kết hợp siêu đồ thị thưa và block hiệu quả"""
    def __init__(self, num_classes=2, in_channels=3, num_joints=17, max_persons=5):
        super().__init__()
        self.max_persons = max_persons
        self.num_joints = num_joints
        self.num_total_joints = num_joints * max_persons
        
        # Xây dựng ma trận kề thưa thông minh
        A = self.build_sparse_adjacency()
        
        # Mạng lõi ST-GCN
        self.stgcn = nn.Sequential(
            EfficientSTGCNBlock(in_channels, 64, A, residual=False),
            EfficientSTGCNBlock(64, 64, A),
            EfficientSTGCNBlock(64, 64, A),
            EfficientSTGCNBlock(64, 128, A, stride=2),
            EfficientSTGCNBlock(128, 128, A),
            EfficientSTGCNBlock(128, 256, A, stride=2),
            EfficientSTGCNBlock(256, 256, A),
        )
        
        # Phân loại
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (N, C, 1, 1)
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def build_sparse_adjacency(self):
        """Xây dựng ma trận kề thưa: chỉ kết nối các khớp quan trọng"""
        # Ma trận kề nội bộ (intra-body) cho 1 người
        intra_edges = [
            (0,1), (0,2), (1,3), (2,4), (5,7), (6,8), (7,9), (8,10),
            (5,11), (6,12), (11,13), (12,14), (13,15), (14,16)
        ]
        A_intra = np.eye(self.num_joints, dtype=np.float32)
        for i, j in intra_edges:
            A_intra[i, j] = 1
            A_intra[j, i] = 1
        
        # Ma trận kề liên người (inter-body): chỉ các khớp tương tác
        interaction_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Khớp trên
        A_inter = np.zeros((self.num_joints, self.num_joints), dtype=np.float32)
        for i in interaction_joints:
            for j in interaction_joints:
                if i != j:
                    A_inter[i, j] = 1  # Đánh dấu có thể tương tác
        
        # Tạo siêu đồ thị cho max_persons người
        A_mega = np.zeros((self.num_total_joints, self.num_total_joints), dtype=np.float32)
        
        # Điền ma trận kề nội bộ cho từng người
        for p in range(self.max_persons):
            start_idx = p * self.num_joints
            A_mega[start_idx:start_idx+self.num_joints, start_idx:start_idx+self.num_joints] = A_intra
        
        # Điền ma trận kề liên người: chỉ giữa các khớp tương tác của người khác nhau
        for p1 in range(self.max_persons):
            for p2 in range(p1+1, self.max_persons):
                start1 = p1 * self.num_joints
                start2 = p2 * self.num_joints
                
                # Chỉ kết nối các khớp trong interaction_joints
                for i in interaction_joints:
                    for j in interaction_joints:
                        A_mega[start1+i, start2+j] = A_inter[i, j]
                        A_mega[start2+j, start1+i] = A_inter[i, j]
        
        # Thêm self-loop
        A_mega += np.eye(self.num_total_joints, dtype=np.float32)
        return torch.from_numpy(A_mega).float()

    def forward(self, keypoints):
        """
        Input: (batch, time, persons, joints, channels)
        - persons có thể thay đổi, nhưng tối đa max_persons
        """
        N, T, P, J, C = keypoints.shape
        
        # Padding động nếu số người ít hơn max_persons
        if P < self.max_persons:
            pad_size = self.max_persons - P
            # Tạo padding tensor (batch, time, pad_size, joints, channels)
            padding = torch.zeros(N, T, pad_size, J, C, device=keypoints.device)
            keypoints = torch.cat([keypoints, padding], dim=2)
        
        # Reshape: (N, T, P, J, C) -> (N, C, T, P*J)
        x = keypoints.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(N, C, T, self.num_total_joints)
        
        # Forward qua mạng
        x = self.stgcn(x)
        return self.fc(x)