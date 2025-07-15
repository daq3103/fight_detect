import os
import cv2
import glob
import pickle
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from configs.configs import parse_arguments # Giả sử bạn đã thêm các arg cần thiết

# --- Khởi tạo MediaPipe Pose ---
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def extract_poses_from_video(video_path, max_persons, num_joints, in_channels):
    """Trích xuất keypoints từ một video, xử lý nhiều người và đệm."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return None

    all_frames_keypoints = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển đổi màu sắc từ BGR sang RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(image_rgb)
        
        # Mặc định là mảng rỗng nếu không phát hiện người nào
        current_frame_keypoints = np.zeros((max_persons, num_joints, in_channels), dtype=np.float32)

        if results.pose_landmarks:
            # Chỉ lấy tối đa `max_persons` người
            num_detected = 1 # MediaPipe v2 chỉ hỗ trợ 1 người, nhưng ta chuẩn bị cho tương lai
            persons_to_process = min(num_detected, max_persons)

            # Lấy keypoints cho người được phát hiện
            landmarks = results.pose_landmarks.landmark
            person_kps = np.array([[lm.x, lm.y, lm.visibility] for lm in landmarks], dtype=np.float32)
            
            # Đảm bảo in_channels là 3
            if in_channels == 2:
                person_kps = person_kps[:, :2]

            current_frame_keypoints[0, :, :] = person_kps

        all_frames_keypoints.append(current_frame_keypoints)

    cap.release()

    if not all_frames_keypoints:
        return None
        
    return np.stack(all_frames_keypoints, axis=0) # Shape: (T, max_persons, J, C)

def process_data(args):
    """
    Hàm chính để xử lý toàn bộ dataset, chia train/val và lưu kết quả.
    """
    print("Bắt đầu quá trình tiền xử lý keypoint...")
    
    # Tạo thư mục lưu trữ nếu chưa có
    os.makedirs(args.data_preprocessed_dir, exist_ok=True)
    
    all_video_paths = []
    all_labels = []

    # 1. Thu thập tất cả các video và nhãn
    for class_index, class_name in enumerate(args.classes_list):
        class_dir = os.path.join(args.data_raw_dir, class_name)
        video_paths = glob.glob(os.path.join(class_dir, "*.mp4"))
        all_video_paths.extend(video_paths)
        all_labels.extend([class_index] * len(video_paths))
        
    print(f"Tìm thấy tổng cộng {len(all_video_paths)} video.")

    # 2. Chia dữ liệu thành tập train và validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_video_paths, all_labels,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=all_labels # Giữ tỷ lệ các lớp trong cả 2 tập
    )
    
    print(f"Chia dữ liệu: {len(train_paths)} train, {len(val_paths)} validation.")

    # 3. Hàm xử lý cho một tập con (train hoặc val)
    def process_subset(paths, labels, subset_name):
        print(f"\n--- Bắt đầu xử lý tập {subset_name} ---")
        keypoints_dict = {}
        labels_list = []
        
        for i, video_path in enumerate(tqdm(paths, desc=f"Processing {subset_name} videos")):
            video_id = os.path.basename(video_path)
            label = labels[i]
            
            # Trích xuất keypoints
            keypoints = extract_poses_from_video(video_path, args.max_persons, args.num_joints, args.in_channels)
            
            if keypoints is not None and len(keypoints) > 0:
                keypoints_dict[video_id] = keypoints
                labels_list.append({'video_id': video_id, 'label': label})
        
        # Lưu kết quả
        data_output_path = os.path.join(args.data_preprocessed_dir, f"{subset_name}_data.npy")
        label_output_path = os.path.join(args.data_preprocessed_dir, f"{subset_name}_labels.pkl")
        
        np.save(data_output_path, keypoints_dict)
        with open(label_output_path, 'wb') as f:
            pickle.dump(labels_list, f)
            
        print(f"Đã lưu dữ liệu keypoint của tập {subset_name} vào {data_output_path}")
        print(f"Đã lưu nhãn của tập {subset_name} vào {label_output_path}")

    # 4. Xử lý cả hai tập
    process_subset(train_paths, train_labels, 'train')
    process_subset(val_paths, val_labels, 'val')
    
    print("\nHoàn tất tiền xử lý dữ liệu keypoint!")

if __name__ == '__main__':
    args = parse_arguments()
    # Hãy đảm bảo file configs.py có các arguments này
    # args.data_raw_dir
    # args.data_preprocessed_dir
    # args.classes_list
    # args.val_split
    # args.seed
    # args.max_persons
    # args.num_joints
    # args.in_channels
    process_data(args)