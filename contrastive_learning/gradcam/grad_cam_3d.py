import numpy as np
import torch
import torch.nn.functional as F

from configs.default_config import (
    STAGE1_CL_CONFIG,
    DEVICE,
    SEQUENCE_LENGTH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)

class GradCAM3D:
    """
    Class để tính toán Grad-CAM cho model CNN 3D.
    Nó sẽ "nhìn" vào feature map của một layer cụ thể.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        # Hook để lấy feature map sau forward pass
        target_layer.register_forward_hook(self._save_feature_maps)
        # Hook để lấy gradient sau backward pass
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor, target_class_idx):
        """
        Args:
            input_tensor (torch.Tensor): Tensor đầu vào cho model, shape [1, C, T, H, W]
            target_class_idx (int): Index của lớp cần visualize

        Returns:
            np.ndarray: Heatmap đã được chuẩn hóa, shape [T, H, W]
        """
        self.model.eval()
        logits = self.model(input_tensor.to(DEVICE))
        
        # Chỉ lấy score của lớp target
        target_score = logits[0, target_class_idx]

        self.model.zero_grad()
        target_score.backward(retain_graph=True)

        # Tính trọng số alpha (Global Average Pooling trên gradient)
        # self.gradients shape: [1, C, T, H, W]
        weights = torch.mean(self.gradients, dim=[2, 3, 4], keepdim=True)
        
        # Tạo heatmap
        # self.feature_maps shape: [1, C, T, H, W]
        weighted_features = self.feature_maps * weights
        heatmap = torch.sum(weighted_features, dim=1).squeeze(0)
        
        # ReLU để chỉ giữ lại các vùng có ảnh hưởng tích cực
        heatmap = F.relu(heatmap)
        
        # Chuẩn hóa heatmap về [0, 1] cho mỗi frame
        heatmap_normalized = []
        for t in range(heatmap.shape[0]):
            frame_heatmap = heatmap[t, :, :]
            # Tránh chia cho 0 nếu heatmap toàn số 0
            if frame_heatmap.max() > 0:
                frame_heatmap = frame_heatmap / frame_heatmap.max()
            heatmap_normalized.append(frame_heatmap.cpu().numpy())
            
        return np.stack(heatmap_normalized)
