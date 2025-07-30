# data/dataset.py
import torch
from torch.utils.data import Dataset
import os
import glob
from data.data_utils import frames_extraction

class SupervisedVideoDataset(Dataset):
    def __init__(self, data_dir, classes_list, sequence_length, image_height, image_width, transform=None):
        self.data_dir = data_dir
        self.classes_list = classes_list
        self.sequence_length = sequence_length
        self.image_height = image_height
        self.image_width = image_width
        self.transform = transform
        
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for class_index, class_name in enumerate(self.classes_list):
            class_path = os.path.join(self.data_dir, class_name)
            # Hỗ trợ nhiều định dạng video
            for ext in ("*.avi", "*.mp4"):
                video_paths = glob.glob(os.path.join(class_path, ext))
                for video_path in video_paths:
                    samples.append((video_path, class_index))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        # Trích xuất frames từ video
        frames = frames_extraction(video_path, self.image_height, self.image_width, self.sequence_length)
        
        # Nếu video không hợp lệ, trả về None
        if frames is None:
            # DataLoader sẽ bỏ qua mẫu này nếu collate_fn được cấu hình đúng
            return None

        # Chuyển đổi (Seq, H, W, C) -> (Seq, C, H, W)
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
        
        if self.transform:
            frames_tensor = self.transform(frames_tensor)
            
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return frames_tensor, label_tensor

# Dataset cho Giai đoạn 1 - Contrastive Learning
class ContrastiveVideoDataset(Dataset):
    def __init__(self, data_dir, sequence_length, image_height, image_width, transform):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.image_height = image_height
        self.image_width = image_width
        # Transform này sẽ được áp dụng 2 lần để tạo 2 view
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        # Quét tất cả video trong thư mục con (Fight, NonFight) mà không cần quan tâm nhãn
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith((".avi", ".mp4")):
                    samples.append(os.path.join(root, file))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path = self.samples[idx]
        frames = frames_extraction(video_path, self.image_height, self.image_width, self.sequence_length)
        
        if frames is None:
            # Trả về None để collate_fn xử lý
            return None

        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
        
        # Tạo 2 view khác nhau từ cùng 1 clip
        view_1 = self.transform(frames_tensor)
        view_2 = self.transform(frames_tensor)
        
        return view_1, view_2

# Hàm collate_fn chung cho cả 2 dataset
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        # Trả về tuple rỗng nếu batch rỗng
        return ()
    return torch.utils.data.dataloader.default_collate(batch)