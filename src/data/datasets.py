# data/datasets.py (phiên bản đã cập nhật)
import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np
import random
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
    Dataset để tải dữ liệu keypoint cho mô hình nhận dạng hành động dựa trên khung xương.

    Args:
        data_path (str): Đường dẫn đến tệp .npy chứa dữ liệu keypoint.
                         Tệp này nên chứa một dictionary-like object (lưu bằng np.save(..., allow_pickle=True))
                         với key là video_id và value là mảng numpy (T, P, J, C).
        label_path (str): Đường dẫn đến tệp .pkl chứa danh sách các nhãn.
                          Mỗi phần tử là một dict {'video_id': str, 'label': int}.
        sequence_length (int): Số lượng khung hình mong muốn cho mỗi mẫu.
        max_persons (int): Số người tối đa được xem xét trong mỗi khung hình.
        num_joints (int): Số lượng khớp của mỗi người.
        in_channels (int): Số kênh cho mỗi khớp (ví dụ: 2 cho (x,y), 3 cho (x,y,score)).
        is_train (bool): Chế độ train (True) sẽ thực hiện lấy mẫu ngẫu nhiên các khung hình.
                         Chế độ eval (False) sẽ lấy mẫu ở trung tâm.
    """
    def __init__(self, data_path, label_path, sequence_length, max_persons, num_joints, in_channels, is_train=True):
        self.data_path = data_path
        self.label_path = label_path
        self.sequence_length = sequence_length
        self.max_persons = max_persons
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.is_train = is_train

        self._load_data()

    def _load_data(self):
        # Tải danh sách các mẫu (video_id và nhãn tương ứng)
        self.sample_list = np.load(self.label_path, allow_pickle=True)
        
        # Tải toàn bộ dữ liệu keypoint vào bộ nhớ.
        # Lưu ý: Nếu dataset quá lớn, bạn có thể cần sử dụng np.load với mmap_mode='r'.
        self.keypoints_data = np.load(self.data_path, allow_pickle=True).item()
        
        print(f"Đã tải {len(self.sample_list)} mẫu từ {self.label_path}")
        print(f"Đã tải dữ liệu keypoint cho {len(self.keypoints_data)} video.")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        # 1. Lấy thông tin mẫu
        sample_info = self.sample_list[index]
        video_id = sample_info['video_id']
        label = sample_info['label']

        # 2. Lấy dữ liệu keypoint từ dictionary đã tải
        keypoints = self.keypoints_data[video_id]  # Shape: (T, P, J, C)

        # 3. Xử lý khung hình (Temporal Sampling)
        total_frames = keypoints.shape[0]
        
        if total_frames > self.sequence_length:
            # Nếu video dài hơn, lấy một đoạn ngẫu nhiên (khi train) hoặc ở giữa (khi val/test)
            if self.is_train:
                start_frame = random.randint(0, total_frames - self.sequence_length)
            else:
                start_frame = (total_frames - self.sequence_length) // 2
            
            keypoints = keypoints[start_frame : start_frame + self.sequence_length]
        
        elif total_frames < self.sequence_length:
            # Nếu video ngắn hơn, đệm (pad) bằng cách lặp lại khung hình cuối cùng
            padding_needed = self.sequence_length - total_frames
            last_frame = keypoints[-1, :, :, :][np.newaxis, ...] # Thêm chiều T
            padding = np.tile(last_frame, (padding_needed, 1, 1, 1))
            keypoints = np.concatenate([keypoints, padding], axis=0)

        # 4. Xử lý số người (Person Padding/Truncating)
        # Shape hiện tại: (sequence_length, P_original, J, C)
        num_persons_original = keypoints.shape[1]
        
        # Tạo một tensor rỗng với kích thước chuẩn (T, max_P, J, C)
        processed_keypoints = np.zeros((self.sequence_length, self.max_persons, self.num_joints, self.in_channels), dtype=np.float32)
        
        # Số người cần xử lý là số nhỏ hơn giữa số người thực tế và max_persons
        num_persons_to_process = min(num_persons_original, self.max_persons)
        
        # Sao chép dữ liệu
        processed_keypoints[:, :num_persons_to_process, :, :] = keypoints[:, :num_persons_to_process, :, :]

        # 5. Chuyển đổi sang Tensor
        keypoints_tensor = torch.from_numpy(processed_keypoints).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Shape cuối cùng của keypoints_tensor: (T, P, J, C)
        # Tuy nhiên, model của bạn yêu cầu (N, C, T, P*J), việc reshape này sẽ được thực hiện trong `forward` của model.
        # Dataset chỉ cần trả về (T, P, J, C) là đủ.
        # Nhưng để phù hợp với code trainer, ta sẽ đổi chiều ngay tại đây.
        # Input cho model: (N, C, T, V) với V = P * J
        # Từ (T, P, J, C) -> (C, T, P, J) -> (C, T, P*J)
        keypoints_tensor = keypoints_tensor.permute(3, 0, 1, 2).contiguous() # (C, T, P, J)
        keypoints_tensor = keypoints_tensor.view(self.in_channels, self.sequence_length, self.max_persons * self.num_joints) # (C, T, V)

        # Model của bạn lại yêu cầu input là (N, T, P, J, C), và reshape bên trong.
        # Vì vậy, chúng ta sẽ trả về shape (T, P, J, C) và để model tự xử lý.
        # Ta sẽ quay lại bước trước khi permute
        keypoints_tensor = torch.from_numpy(processed_keypoints).float() # (T, P, J, C)

        # **LƯU Ý QUAN TRỌNG:**
        # Model `OptimizedFightDetector` của bạn có phần `forward` như sau:
        # Input: (batch, time, persons, joints, channels)
        # x = keypoints.permute(0, 4, 1, 2, 3).contiguous() # (N, C, T, P, J)
        # x = x.view(N, C, T, self.num_total_joints) # (N, C, T, P*J)
        # Dataloader sẽ thêm chiều `batch (N)` vào đầu.
        # Vì vậy, `__getitem__` chỉ cần trả về (time, persons, joints, channels) là chính xác.
        
        return keypoints_tensor, label_tensor
