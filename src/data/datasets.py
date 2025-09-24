# data/datasets.py (phiên bản đã cập nhật)
import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np
import random
import pickle
import cv2 as cv
from ultralytics import YOLO

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
        
        # File .npy là một mảng NumPy lớn với shape (N, T, P, J, C)
        self.keypoints_data = np.load(self.data_path)
        
        with open(self.label_path, 'rb') as f:
            self.labels = pickle.load(f)
        
        print(f"Đã tải thành công: {self.keypoints_data.shape[0]} mẫu.")

    def __len__(self):
        """Trả về tổng số mẫu trong dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Lấy một mẫu dữ liệu tại vị trí idx."""
        # Lấy dữ liệu keypoints và nhãn tương ứng
        # Shape từ file npy: (T, P, J, C)
        keypoints = self.keypoints_data[idx]
        label = self.labels[idx]

        # Chuyển đổi sang PyTorch tensor
        keypoints_tensor = torch.from_numpy(keypoints).float()

        # Giữ nguyên thứ tự axes (T, P, J, C)
        # Trainer/model sẽ tự permute thành (N, C, T, P, J)

        # Đảm bảo luôn return tuple (tensor, label)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return keypoints_tensor, label_tensor

class SegmentationDataset(Dataset):
    """
    Dataset để xử lý segmentation từ video files và trích xuất các vùng đã được segment.
    """
    def __init__(self, video_dir, model_path="yolo11n-seg.pt", target_class="person", 
                 sequence_length=30, image_height=224, image_width=224):
        """
        Args:
            video_dir: Đường dẫn đến thư mục chứa video files
            model_path: Đường dẫn đến YOLO model
            target_class: Class cần segment (mặc định "person")
            sequence_length: Số frames cần trích xuất từ mỗi video
            image_height: Chiều cao của frame output
            image_width: Chiều rộng của frame output
        """
        self.video_dir = video_dir
        self.target_class = target_class
        self.sequence_length = sequence_length
        self.image_height = image_height
        self.image_width = image_width
        
        # Load YOLO model
        self.model = YOLO(model_path)
        
        # Tìm tất cả video files
        self.video_files = self._get_video_files()
        
    def _get_video_files(self):
        """Lấy danh sách tất cả video files."""
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(self.video_dir, ext)))
        return video_files
    
    def _extract_segmented_frames(self, video_path):
        """
        Trích xuất các frames đã được segment từ video.
        
        Returns:
            segmented_frames: List các frames đã được segment
        """
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return []
        
        segmented_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame to target size
            frame = cv.resize(frame, (self.image_width, self.image_height))
            
            # Run YOLO segmentation
            results = self.model(frame, imgsz=640, verbose=False)
            result = results[0]
            
            # Tạo một ảnh đen có cùng kích thước với frame
            segmented_frame = np.zeros_like(frame)
            
            # Check if masks exist
            if result.masks is not None and result.boxes is not None:
                # Lấy tất cả các vùng đã được segment
                for i, mask in enumerate(result.masks.data):
                    cls_idx = int(result.boxes.cls[i])
                    
                    # Chỉ lấy class mong muốn
                    if result.names[cls_idx] == self.target_class:
                        mask_np = mask.cpu().numpy()
                        # Resize mask to match frame shape
                        mask_resized = cv.resize(mask_np, (frame.shape[1], frame.shape[0]), 
                                               interpolation=cv.INTER_NEAREST)
                        # Copy pixel từ frame gốc vào vùng có mask
                        segmented_frame[mask_resized > 0.5] = frame[mask_resized > 0.5]
            
            # Chỉ thêm frame nếu có segmentation
            if np.any(segmented_frame):
                segmented_frames.append(segmented_frame)
        
        cap.release()
        return segmented_frames
    
    def _sample_frames(self, frames):
        """
        Sample frames để đảm bảo có đúng sequence_length frames.
        """
        if len(frames) == 0:
            # Trả về frames đen nếu không có frames nào
            return [np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8) 
                   for _ in range(self.sequence_length)]
        
        if len(frames) >= self.sequence_length:
            # Random sample nếu có nhiều frames
            indices = np.random.choice(len(frames), self.sequence_length, replace=False)
            indices = np.sort(indices)
            return [frames[i] for i in indices]
        else:
            # Repeat frames nếu không đủ
            sampled = frames.copy()
            while len(sampled) < self.sequence_length:
                sampled.extend(frames)
            return sampled[:self.sequence_length]
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        """
        Lấy một sequence frames đã được segment từ video tại index idx.
        
        Returns:
            frames_tensor: Tensor shape (T, C, H, W)
            video_path: Đường dẫn video (có thể dùng làm label hoặc identifier)
        """
        video_path = self.video_files[idx]
        
        # Extract segmented frames
        segmented_frames = self._extract_segmented_frames(video_path)
        
        # Sample frames to get exact sequence_length
        sampled_frames = self._sample_frames(segmented_frames)
        
        # Convert to numpy array
        frames_array = np.array(sampled_frames)  # Shape: (T, H, W, C)
        
        # Normalize pixel values to [0, 1]
        frames_array = frames_array.astype(np.float32) / 255.0
        
        # Convert to tensor and change dimension order to (T, C, H, W)
        frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2)
        
        return frames_tensor, video_path


class SegmentationProcessor:
    """
    Class utility để xử lý segmentation từ một video file.
    Sử dụng khi chỉ cần xử lý một video duy nhất.
    """
    def __init__(self, model_path="yolo11n-seg.pt", target_class="person"):
        self.model = YOLO(model_path)
        self.target_class = target_class
    
    def process_video(self, video_path, output_path=None, display=True):
        """
        Xử lý một video và hiển thị hoặc lưu kết quả.
        
        Args:
            video_path: Đường dẫn đến video input
            output_path: Đường dẫn để lưu video output (optional)
            display: Có hiển thị video trong quá trình xử lý không
        """
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return
        
        # Get video properties for output
        fps = int(cap.get(cv.CAP_PROP_FPS))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            writer = cv.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            results = self.model(frame, imgsz=640, verbose=False)
            result = results[0]
            
            # Tạo một ảnh đen có cùng kích thước với frame gốc
            segmented_frame = np.zeros_like(frame)
            
            # Check if masks exist
            if result.masks is not None and result.boxes is not None:
                # Lấy tất cả các vùng đã được segment
                for i, mask in enumerate(result.masks.data):
                    cls_idx = int(result.boxes.cls[i])
                    
                    # Chỉ lấy class mong muốn
                    if result.names[cls_idx] == self.target_class:
                        mask_np = mask.cpu().numpy()
                        # Resize mask to match frame shape
                        mask_resized = cv.resize(mask_np, (frame.shape[1], frame.shape[0]), 
                                               interpolation=cv.INTER_NEAREST)
                        # Copy pixel từ frame gốc vào vùng có mask
                        segmented_frame[mask_resized > 0.5] = frame[mask_resized > 0.5]
            
            # Save frame if writer is available
            if writer:
                writer.write(segmented_frame)
            
            # Display frame if requested
            if display:
                cv.imshow('Segmented Frame', segmented_frame)
                if cv.waitKey(1) == ord('q'):
                    break
        
        cap.release()
        if writer:
            writer.release()
        if display:
            cv.destroyAllWindows()

