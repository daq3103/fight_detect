# data/dataset.py
import torch
from torch.utils.data import Dataset
import os
import glob
from data.data_utils import frames_extraction
import random

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

class SemanticContrastiveDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 sequence_length, 
                 image_height, 
                 image_width, 
                 transform=None,
                 classes=['Fight', 'NonFight']):
        """
        Dataset cho Contrastive Learning sử dụng semantic labels
        
        Args:
            data_dir: Thư mục chứa dữ liệu (có cấu trúc con /Fight, /NonFight)
            sequence_length: Số frame mỗi video
            image_height, image_width: Kích thước ảnh
            transform: Augmentation áp dụng
            classes: Danh sách tên lớp
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.image_height = image_height
        self.image_width = image_width
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
        # Tải danh sách video theo lớp
        self.samples = self._load_samples()
        self.class_indices = self._build_class_indices()

    def _load_samples(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            for ext in ("*.avi", "*.mp4"):
                video_paths = glob.glob(os.path.join(class_dir, ext))
                for video_path in video_paths:
                    samples.append({
                        'path': video_path,
                        'class': self.class_to_idx[class_name]
                    })
        return samples

    def _build_class_indices(self):
        """Xây dựng index cho từng lớp để lấy mẫu nhanh"""
        class_indices = {i: [] for i in range(len(self.classes))}
        for idx, sample in enumerate(self.samples):
            class_indices[sample['class']].append(idx)
        return class_indices

    def __len__(self):
        return len(self.samples)

    # def __getitem__(self, idx):
    #     # Anchor video
    #     anchor = self.samples[idx]
    #     anchor_frames = self._load_and_transform(anchor['path'])
        
    #     # Tạo positive pair: Cùng lớp nhưng khác video
    #     pos_idx = self._get_positive_index(idx, anchor['class'])
    #     pos_frames = self._load_and_transform(self.samples[pos_idx]['path'])
        
    #     # Tạo negative pair: Khác lớp
    #     neg_idx = self._get_negative_index(anchor['class'])
    #     neg_frames = self._load_and_transform(self.samples[neg_idx]['path'])
        
    #     return {
    #         'anchor': anchor_frames,
    #         'positive': pos_frames,
    #         'negative': neg_frames,
    #         'class': anchor['class']
    #     }

    def __getitem__(self, idx):
        # 1. Lấy thông tin video anchor
        anchor_info = self.samples[idx]
        
        # 2. Tải frames của video anchor MỘT LẦN
        # Tạm thời chưa transform ở bước này
        frames = frames_extraction(
            anchor_info['path'], 
            self.image_height, 
            self.image_width, 
            self.sequence_length
        )
        
        # Kiểm tra nếu tải video lỗi
        if frames is None:
            # Ta sẽ lọc các sample None này trong collate_fn
            return None

        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()

        # 3. Tạo positive pair bằng cách augment 2 lần
        # Đây là thay đổi cốt lõi!
        if self.transform:
            anchor_frames = self.transform(frames_tensor)
            positive_frames = self.transform(frames_tensor)
        else:
            anchor_frames = frames_tensor
            positive_frames = frames_tensor
        
        # 4. Lấy negative từ một lớp khác (giữ nguyên logic cũ)
        neg_idx = self._get_negative_index(anchor_info['class'])
        # Cần một vòng lặp phòng trường hợp video negative bị lỗi
        neg_frames = None
        while neg_frames is None:
            neg_frames = self._load_and_transform(self.samples[neg_idx]['path']) # Dùng lại hàm cũ
            if neg_frames is None:
                neg_idx = self._get_negative_index(anchor_info['class'])

        return {
            'anchor': anchor_frames,
            'positive': positive_frames,
            'negative': neg_frames,
            'class': anchor_info['class']
        }
    
    # Hàm _load_and_transform giờ chỉ cần cho negative
    def _load_and_transform(self, video_path):
        frames = frames_extraction(
            video_path, 
            self.image_height, 
            self.image_width, 
            self.sequence_length
        )
        if frames is None:
            return None
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        if self.transform:
            frames_tensor = self.transform(frames_tensor)
        return frames_tensor

    def _load_and_transform(self, video_path):
        """Trích xuất frames và áp dụng transform"""
        frames = frames_extraction(
            video_path, 
            self.image_height, 
            self.image_width, 
            self.sequence_length
        )
        if frames is None:
            return None
            
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        if self.transform:
            frames_tensor = self.transform(frames_tensor)
        return frames_tensor

    def _get_positive_index(self, anchor_idx, class_label):
        """Lấy ngẫu nhiên 1 video cùng lớp (khác anchor)"""
        indices = self.class_indices[class_label]
        indices = [i for i in indices if i != anchor_idx]
        return random.choice(indices) if indices else anchor_idx

    def _get_negative_index(self, class_label):
        """Lấy ngẫu nhiên 1 video từ lớp khác"""
        negative_classes = [i for i in range(len(self.classes)) if i != class_label]
        target_class = random.choice(negative_classes)
        return random.choice(self.class_indices[target_class])

def semantic_collate_fn(batch):
    batch = [b for b in batch if b is not None and 
             b['anchor'] is not None and 
             b['positive'] is not None and 
             b['negative'] is not None]
    
    if not batch:
        return ()
    
    anchors = torch.stack([b['anchor'] for b in batch])
    positives = torch.stack([b['positive'] for b in batch])
    negatives = torch.stack([b['negative'] for b in batch])
    classes = torch.tensor([b['class'] for b in batch], dtype=torch.long)
    
    return {
        'anchors': anchors,
        'positives': positives,
        'negatives': negatives,
        'classes': classes
    }


def collate_fn(batch):

    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else (torch.tensor([]), torch.tensor([]))