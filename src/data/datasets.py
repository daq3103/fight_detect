# data/datasets.py (phiên bản đã cập nhật)
import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np
import random
import pickle
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
    

class PoseDataset(Dataset):
    """
    Dataset để tải dữ liệu keypoints đã được gộp lại từ file .npy.
    
    Phiên bản này được sửa để đọc trực tiếp file .npy chứa một mảng NumPy lớn,
    thay vì đọc một đối tượng được đóng gói.
    """
    def __init__(self, data_path, label_path, sequence_length, num_joints, max_persons, in_channels, is_train=False):
        self.data_path = data_path
        self.label_path = label_path
        self.is_train = is_train
        
        # Lưu lại các tham số cấu trúc dữ liệu
        self.sequence_length = sequence_length
        self.num_joints = num_joints
        self.max_persons = max_persons
        self.in_channels = in_channels
        
        self.keypoints_data = None
        self.labels = None
        self._load_data()

    def _load_data(self):
        """Tải dữ liệu keypoints và nhãn từ các file đã được aggregate."""
        print(f"Đang tải dữ liệu từ: {self.data_path}")
        
        # === SỬA LỖI TẠI ĐÂY ===
        # File .npy của bạn là một mảng NumPy lớn, vì vậy chúng ta chỉ cần
        # dùng np.load() để đọc nó trực tiếp. Bỏ `.item()` đi.
        self.keypoints_data = np.load(self.data_path)
        
        with open(self.label_path, 'rb') as f:
            self.labels = pickle.load(f)
        
        print(f"Đã tải thành công: {self.keypoints_data.shape[0]} mẫu.")
        # Shape mong đợi: (Số mẫu, Sequence Length, Max Persons, Num Joints, Channels)
        # Ví dụ: (500, 30, 2, 17, 3)

    def __len__(self):
        """Trả về tổng số mẫu trong dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Lấy một mẫu dữ liệu tại vị trí idx."""
        # Lấy dữ liệu keypoints và nhãn tương ứng
        # Shape ban đầu từ file npy: (T, P, J, C)
        keypoints = self.keypoints_data[idx] 
        label = self.labels[idx]

        # Chuyển đổi sang PyTorch tensor
        keypoints_tensor = torch.from_numpy(keypoints).float()
        
        # PyTorch và các model 3D-CNN thường mong muốn channel ở trước: (C, T, P, J)
        # T: time (sequence_length), P: person, J: joint, C: channel (x, y, conf)
        keypoints_tensor = keypoints_tensor.permute(3, 0, 1, 2) 

        label_tensor = torch.tensor(label, dtype=torch.long)

        return keypoints_tensor, label_tensor