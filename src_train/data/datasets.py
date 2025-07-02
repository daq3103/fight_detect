# data/datasets.py (phiên bản đã cập nhật)
import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np

# KHÔNG CẦN import frames_extraction ở đây nữa

class VideoDataset(Dataset):
    def __init__(self, data_dir, classes_list, sequence_length, image_height, image_width):
        self.data_dir = data_dir # Đây là đường dẫn đến thư mục PROCESSED_DATA
        self.classes_list = classes_list
        self.sequence_length = sequence_length
        self.image_height = image_height
        self.image_width = image_width
        
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for class_index, class_name in enumerate(self.classes_list):
            class_path = os.path.join(self.data_dir, class_name)
            # Bây giờ chúng ta tìm các file .npy thay vì .mp4
            file_paths = glob.glob(os.path.join(class_path, "*.npy"))
            for file_path in file_paths:
                samples.append((file_path, class_index))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        # *** THAY ĐỔI QUAN TRỌNG NHẤT LÀ Ở ĐÂY ***
        # Thay vì gọi frames_extraction, chúng ta chỉ cần tải file .npy
        frames = np.load(file_path)
        
        # Chuyển sang Tensor
        # Chuyển đổi (Seq, H, W, C) -> (Seq, C, H, W) là một thực hành tốt cho PyTorch
        frames_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return frames_tensor, label_tensor