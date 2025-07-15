import os
import cv2
import glob
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from configs.configs import parse_arguments 

# --- Import YOLO-Pose specific libraries ---
import torch

def extract_poses_from_video(video_path, max_persons, num_joints, in_channels, pose_detector_model):
    """Trích xuất keypoints từ một video sử dụng YOLO-Pose, xử lý nhiều người và đệm."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return None

    all_frames_keypoints = []
    
    YOLO_POSE_NUM_JOINTS = 17 
    if num_joints != YOLO_POSE_NUM_JOINTS:
        print(f"Cảnh báo: num_joints trong cấu hình ({num_joints}) không khớp với số khớp của YOLO-Pose ({YOLO_POSE_NUM_JOINTS}).")
        print(f"Sẽ sử dụng {YOLO_POSE_NUM_JOINTS} khớp của YOLO-Pose.")
        num_joints_to_use = YOLO_POSE_NUM_JOINTS
    else:
        num_joints_to_use = num_joints

    img_size = 640 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w = frame.shape[:2]
        img_input = cv2.resize(frame, (img_size, img_size))
        img_input = img_input[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img_input = np.ascontiguousarray(img_input)
        img_input = torch.from_numpy(img_input).to(pose_detector_model.device)
        img_input = img_input.float() / 255.0
        img_input = img_input.unsqueeze(0)

        with torch.no_grad():
            results = pose_detector_model(img_input)[0] # Giả định output là tensor đầu tiên

        current_frame_keypoints = np.zeros((max_persons, num_joints_to_use, in_channels), dtype=np.float32)
        person_count = 0

        if results is not None and len(results) > 0:
            # Lặp qua từng phát hiện. Dạng của `results` là tensor
            # Mỗi hàng trong `results` là một phát hiện [bbox_coords, conf, class_id, keypoints_data]
            # keypoints_data là một mảng phẳng (x1, y1, conf1, x2, y2, conf2, ...)
            
            for detection in results.cpu().numpy():
                # Lấy bbox (4 giá trị), conf (1 giá trị), class_id (1 giá trị)
                # và phần còn lại là keypoints_data
                bbox = detection[:4]
                conf = detection[4]
                cls = detection[5]
                kps_flat_data = detection[6:] # Đây là phần keypoints_data

                if int(cls) == 0 and conf > 0.5:
                    if person_count >= max_persons:
                        break

                    # Đảm bảo kps_flat_data có đủ số phần tử để reshape
                    if len(kps_flat_data) == num_joints_to_use * 3:
                        kps = kps_flat_data.reshape(-1, 3) # Chuyển thành (17, 3)
                    else:
                        print(f"Cảnh báo: Dữ liệu keypoint không đủ phần tử để định hình lại (mong đợi {num_joints_to_use * 3}, nhận được {len(kps_flat_data)}). Bỏ qua phát hiện này.")
                        continue # Bỏ qua phát hiện này và chuyển sang phát hiện tiếp theo

                    # Scale keypoints trở lại kích thước ảnh gốc
                    kps[:, 0] = kps[:, 0] * (img_w / img_size)
                    kps[:, 1] = kps[:, 1] * (img_h / img_size)
                    
                    person_kps = np.array(kps, dtype=np.float32)

                    if in_channels == 2:
                        person_kps = person_kps[:, :2]
                    elif in_channels == 3:
                        person_kps = person_kps

                    if person_kps.shape[0] == num_joints_to_use:
                        current_frame_keypoints[person_count, :, :] = person_kps
                        person_count += 1
                    else:
                        print(f"Cảnh báo: Số khớp được xử lý ({person_kps.shape[0]}) không khớp với số khớp dự kiến ({num_joints_to_use}). Bỏ qua phát hiện này.")

        all_frames_keypoints.append(current_frame_keypoints)

    cap.release()

    if not all_frames_keypoints:
        return None
        
    return np.stack(all_frames_keypoints, axis=0)

def process_data(args, pose_detector_model):
    """
    Hàm chính để xử lý toàn bộ dataset, chia train/val và lưu kết quả.
    """
    print("Bắt đầu quá trình tiền xử lý keypoint...")
    
    os.makedirs(args.data_preprocessed_dir, exist_ok=True)
    
    all_video_paths = []
    all_labels = []

    for class_index, class_name in enumerate(args.classes_list):
        class_dir = os.path.join(args.data_raw_dir, class_name)
        video_paths = glob.glob(os.path.join(class_dir, "*.mp4"))
        all_video_paths.extend(video_paths)
        all_labels.extend([class_index] * len(video_paths))
        
    print(f"Tìm thấy tổng cộng {len(all_video_paths)} video.")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_video_paths, all_labels,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=all_labels 
    )
    
    print(f"Chia dữ liệu: {len(train_paths)} train, {len(val_paths)} validation.")

    def process_subset(paths, labels, subset_name):
        print(f"\n--- Bắt đầu xử lý tập {subset_name} ---")
        keypoints_dict = {}
        labels_list = []
        
        for i, video_path in enumerate(tqdm(paths, desc=f"Processing {subset_name} videos")):
            video_id = os.path.basename(video_path)
            label = labels[i]
            
            keypoints = extract_poses_from_video(video_path, args.max_persons, args.num_joints, args.in_channels, pose_detector_model)
            
            if keypoints is not None and len(keypoints) > 0:
                keypoints_dict[video_id] = keypoints
                labels_list.append({'video_id': video_id, 'label': label})
        
        data_output_path = os.path.join(args.data_preprocessed_dir, f"{subset_name}_data.npy")
        label_output_path = os.path.join(args.data_preprocessed_dir, f"{subset_name}_labels.pkl")
        
        np.save(data_output_path, keypoints_dict)
        with open(label_output_path, 'wb') as f:
            pickle.dump(labels_list, f)
            
        print(f"Đã lưu dữ liệu keypoint của tập {subset_name} vào {data_output_path}")
        print(f"Đã lưu nhãn của tập {subset_name} vào {label_output_path}")

    process_subset(train_paths, train_labels, 'train')
    process_subset(val_paths, val_labels, 'val')
    
    print("\nHoàn tất tiền xử lý dữ liệu keypoint!")

def load_yolopose_model(weights_path, config_path=None):
    try:
        import torch
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Sử dụng thiết bị: {device}")

        # --- MOCK MODEL CHO MỤC ĐÍCH TEST CẤU TRÚC (ĐÃ SỬA) ---
        class MockYOLOPoseModel:
            def __init__(self, device):
                self.device = device
                print(f"Mô hình YOLO-Pose giả lập đã được khởi tạo trên {device}.")
            
            def __call__(self, img_tensor):
                batch_size, _, img_h, img_w = img_tensor.shape
                all_detections = []
                
                # Giả lập đầu ra của YOLO-Pose:
                # Mỗi hàng là một phát hiện: [x1, y1, x2, y2, conf, class_id, kp1_x, kp1_y, kp1_conf, ..., kp17_x, kp17_y, kp17_conf]
                # Tổng cộng 4 + 1 + 1 + (17 * 3) = 57 giá trị cho mỗi phát hiện.
                num_kp_values = 17 * 3 # 51 giá trị cho keypoints

                for b in range(batch_size):
                    num_dummy_persons = np.random.randint(1, 3) # Giả lập 1 hoặc 2 người
                    dummy_results_list = []
                    for _ in range(num_dummy_persons):
                        # Bbox: [x1,y1,x2,y2]
                        x1, y1 = np.random.randint(0, img_w // 2 - 10), np.random.randint(0, img_h // 2 - 10)
                        x2, y2 = np.random.randint(x1 + 20, img_w), np.random.randint(y1 + 20, img_h)
                        bbox = np.array([x1, y1, x2, y2])
                        conf = np.random.rand() * 0.3 + 0.6 # conf ~ [0.6, 0.9]
                        cls_id = 0 # person

                        # Keypoints: 17 * 3 (x,y,conf)
                        kps_data = np.random.rand(17, 3).astype(np.float32)
                        # Scale keypoints relative to bbox for realism
                        kps_data[:, 0] = kps_data[:, 0] * (x2 - x1) + x1
                        kps_data[:, 1] = kps_data[:, 1] * (y2 - y1) + y1
                        kps_data[:, 2] = kps_data[:, 2] * 0.5 + 0.5 # conf ~ [0.5, 1.0]

                        # Nối tất cả lại thành một mảng 1D cho mỗi detection
                        detection = np.concatenate((bbox, [conf, cls_id], kps_data.flatten()))
                        dummy_results_list.append(detection)
                    
                    if len(dummy_results_list) > 0:
                        all_detections.append(torch.tensor(np.array(dummy_results_list), dtype=torch.float32).to(self.device))
                    else:
                        all_detections.append(torch.empty((0, 5 + num_kp_values), dtype=torch.float32).to(self.device)) # Empty tensor
                
                return all_detections

        model = MockYOLOPoseModel(device)
        return model, device

    except ImportError as e:
        print(f"Lỗi import thư viện YOLO-Pose: {e}")
        print("Đảm bảo bạn đã cài đặt PyTorch và các dependency của YOLO-Pose 11m.")
        print("Ví dụ: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("Và clone/cài đặt YOLO-Pose 11m theo hướng dẫn của nhà phát triển.")
        return None, None
    except Exception as e:
        print(f"Lỗi khi tải hoặc khởi tạo mô hình YOLO-Pose: {e}")
        return None, None


if __name__ == '__main__':
    args = parse_arguments()
    
    # Cấu hình các tham số cần thiết cho YOLO-Pose
    args.max_persons = 2 
    args.num_joints = 17 
    args.in_channels = 3 

    pose_model, device = load_yolopose_model(weights_path='dummy_yolopose_11m.pt', config_path='dummy_config.yaml')

    if pose_model is not None:
        process_data(args, pose_model)
    else:
        print("Không thể tải mô hình YOLO-Pose. Vui lòng kiểm tra đường dẫn và cài đặt.")