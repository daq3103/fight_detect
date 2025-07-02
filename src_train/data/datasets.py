# data/datasets.py
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm # Để hiển thị thanh tiến trình khi tạo dataset

from data.data_utils import frames_extraction

class VideoDataset(Dataset):
    def __init__(self, data_dir, classes_list, image_height, image_width, sequence_length, transform=None):
        self.data_dir = data_dir
        self.classes_list = classes_list
        self.image_height = image_height
        self.image_width = image_width
        self.sequence_length = sequence_length
        self.transform = transform
        self.video_info = self._collect_video_paths()

    def _collect_video_paths(self):
        """
        Collects all valid video file paths and their corresponding labels.
        This runs once during dataset initialization.
        """
        video_info_list = []
        print("Collecting video paths and checking frame counts...")
        for class_index, class_name in enumerate(self.classes_list):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue

            # Iterate through files with tqdm for progress feedback
            for file_name in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}"):
                video_file_path = os.path.join(class_dir, file_name)
                
                # We need to quickly check if the video is valid and has enough frames
                # without fully extracting all frames, as this can be slow.
                # A more efficient approach for large datasets might be to pre-filter
                # or store metadata. For now, we'll do a quick check.
                
                # OPTIONAL: A more robust check might involve cv2.VideoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
                # and comparing it to self.sequence_length * min_skip_frames_window.
                # However, for simplicity and matching original logic, we'll assume
                # frames_extraction handles the length check within __getitem__.
                
                video_info_list.append({
                    'path': video_file_path,
                    'label': class_index
                })
        print(f"Collected {len(video_info_list)} video paths.")
        return video_info_list

    def __len__(self):
        return len(self.video_info)

    def __getitem__(self, idx):
        video_path = self.video_info[idx]['path']
        label = self.video_info[idx]['label']

        # Extract frames only when an item is requested
        frames = frames_extraction(video_path, self.image_height, self.image_width, self.sequence_length)
        
        # Handle cases where frames_extraction failed or returned fewer frames
        if len(frames) == 0:
            # This is a fallback. In a real scenario, you might want to skip this sample
            # or use a default dummy tensor, but this would break batching if not handled carefully.
            # For simplicity, if frames_extraction returns an empty list, we'll try to find another valid sample.
            # A better approach for real data is to pre-filter invalid videos during _collect_video_paths.
            print(f"Skipping invalid video: {video_path}. Trying next sample.")
            # This recursive call can lead to infinite loop if too many invalid videos
            # A more robust solution would be to filter invalid videos during dataset initialization.
            return self.__getitem__(torch.randint(0, len(self), (1,)).item()) 


        # PyTorch expects (C, H, W) for images, and (batch_size, sequence_length, C, H, W) for video
        # frames_extraction returns (sequence_length, H, W, C)
        # We need to permute to (sequence_length, C, H, W)
        frames_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2)
        
        if self.transform:
            frames_tensor = self.transform(frames_tensor)

        return frames_tensor, label