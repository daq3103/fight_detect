import os
import cv2
import glob
import pickle
import numpy as np
# import mediapipe as mp # Không cần MediaPipe nữa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from configs.configs import parse_arguments 

# --- Import YOLO-Pose specific libraries ---
# Bạn cần cài đặt và cấu hình thư viện YOLO-Pose của mình ở đây
# Đây chỉ là ví dụ giả định, cần thay thế bằng code thực tế của YOLO-Pose 11m
import torch
# Giả sử bạn có thư mục 'yolov7' trong project của mình và các module cần thiết
# import sys
# sys.path.append('path/to/your/yolov7-repo') # Thay bằng đường dẫn thực tế đến repo YOLOv7
# from models.yolo import Model
# from utils.general import non_max_suppression, scale_coords
# from utils.plots import colors, plot_one_box

# Thay thế bằng code tải mô hình YOLO-Pose 11m của bạn
# global pose_detector, device # Khai báo biến global nếu khởi tạo ở ngoài hàm
# # Cấu hình thiết bị (CPU/GPU)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # Tải mô hình YOLO-Pose 11m (ví dụ)
# # Bạn cần tải trọng số và file cấu hình của YOLO-Pose 11m
# # Ví dụ: model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# # Đối với YOLO-Pose 11m, có thể phức tạp hơn, cần clone repo và tải weights
# # model_weights_path = 'path/to/your/yolopose_11m.pt' 
# # model_config_path = 'path/to/your/yolopose_11m.yaml' # Nếu cần
# # pose_detector = Model(model_config_path).to(device)
# # pose_detector.load_state_dict(torch.load(model_weights_path, map_location=device)['model'].state_dict())
# # pose_detector.eval()

# # Giả định khởi tạo đơn giản cho mục đích ví dụ.
# # THỰC TẾ: Bạn phải thay thế phần này bằng cách tải YOLO-Pose 11m của bạn.
# # Có thể sử dụng một mock object hoặc tải một mô hình YOLO đơn giản để test cấu trúc.
# class MockYOLOPoseModel:
#     def __init__(self):
#         print("Mô hình YOLO-Pose giả lập đã được khởi tạo.")
#     
#     def __call__(self, img):
#         # Giả lập đầu ra của YOLO-Pose: list of detections
#         # Mỗi detection: [x1, y1, x2, y2, conf, class_id, keypoints_x, keypoints_y, keypoints_conf]
#         # Keypoints: (17 * 3) -> [x1,y1,conf1, x2,y2,conf2, ..., x17,y17,conf17]
#         # Giả sử phát hiện 1 người với 17 keypoint ngẫu nhiên
#         num_dummy_persons = 1 # Hoặc nhiều hơn nếu muốn test max_persons > 1
#         dummy_results = []
#         for _ in range(num_dummy_persons):
#             dummy_bbox = np.array([100, 100, 200, 200, 0.9, 0], dtype=np.float32) # x1,y1,x2,y2,conf,class_id
#             dummy_kps = np.random.rand(17, 3).astype(np.float32) * 50 + 100 # x,y,conf cho 17 khớp
#             dummy_detection = np.concatenate((dummy_bbox, dummy_kps.flatten()))
#             dummy_results.append(dummy_detection)
#         
#         # Đầu ra của YOLO-Pose thường là một tensor, không phải list.
#         # Cần mô phỏng cấu trúc đầu ra thực tế của YOLO-Pose
#         # Giả sử model.forward(img) trả về một tensor như sau:
#         # [batch_idx, x1, y1, x2, y2, conf, class_id, kp1_x, kp1_y, kp1_conf, ...]
#         # Với YOLOv7-pose, output có thể là (x,y,w,h,conf,class) + (kpx, kpy, kpconf) * num_kps
#         
#         # Đây là một giả định rất đơn giản. Bạn cần điều chỉnh nó để phù hợp với output thực tế của YOLO-Pose.
#         # Có thể YOLO-Pose trả về dict hoặc list of tensors.
#         return [torch.tensor(dummy_results).unsqueeze(0)] # Batch dimension

# pose_detector = MockYOLOPoseModel() # Dùng mô hình giả lập để phát triển cấu trúc hàm trước

# Định nghĩa hàm tải và khởi tạo YOLO-Pose 11m của bạn ở đây.
# Ví dụ:
# def load_yolopose_model(weights_path, config_path=None):
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     # Thay thế bằng logic tải YOLO-Pose 11m thực tế
#     # Ví dụ: model = YOLO_Pose_Model_Class(config_path).to(device)
#     # model.load_state_dict(torch.load(weights_path, map_location=device)['model'].state_dict())
#     # model.eval()
#     # return model, device
#     class MockYOLOPoseModel:
#         def __init__(self):
#             print("Mô hình YOLO-Pose giả lập đã được khởi tạo.")
        
#         def __call__(self, img):
#             # Giả lập đầu ra của YOLO-Pose: list of detections
#             # Mỗi detection: [x1, y1, x2, y2, conf, class_id, keypoints_x, keypoints_y, keypoints_conf]
#             # Keypoints: (17 * 3) -> [x1,y1,conf1, x2,y2,conf2, ..., x17,y17,conf17]
#             num_dummy_persons = 1 
#             dummy_results_list = []
#             for _ in range(num_dummy_persons):
#                 # Giả lập detection bao gồm bbox và 17 keypoints
#                 # [x1,y1,x2,y2,conf,class, kp1x,kp1y,kp1conf, ..., kp17x,kp17y,kp17conf]
#                 dummy_detection = np.random.rand(5 + 17*3).astype(np.float32)
#                 dummy_detection[4] = 0.9 # conf
#                 dummy_detection[5] = 0 # class_id (person)
#                 # Scale dummy bbox to reasonable values
#                 dummy_detection[0:4] = np.random.randint(50, 400, size=4) 
#                 
#                 # Scale dummy keypoints to reasonable values relative to bbox
#                 for i in range(17):
#                     dummy_detection[6 + i*3] = np.random.rand() * (dummy_detection[2] - dummy_detection[0]) + dummy_detection[0]
#                     dummy_detection[6 + i*3 + 1] = np.random.rand() * (dummy_detection[3] - dummy_detection[1]) + dummy_detection[1]
#                     dummy_detection[6 + i*3 + 2] = np.random.rand() * 0.5 + 0.5 # confidence
                    
#                 dummy_results_list.append(dummy_detection)
            
#             if len(dummy_results_list) > 0:
#                 return [torch.tensor(np.array(dummy_results_list)).unsqueeze(0)] # Return as a list of tensor (batch)
#             return []

#     return MockYOLOPoseModel(), torch.device('cpu') # Trả về mô hình giả lập và device

# pose_detector, device = load_yolopose_model('dummy_path.pt') # Gọi hàm tải mô hình của bạn

def extract_poses_from_video(video_path, max_persons, num_joints, in_channels, pose_detector_model): # Truyền model vào
    """Trích xuất keypoints từ một video sử dụng YOLO-Pose, xử lý nhiều người và đệm."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return None

    all_frames_keypoints = []
    
    # YOLO-Pose (COCO) thường có 17 khớp
    YOLO_POSE_NUM_JOINTS = 17 
    if num_joints != YOLO_POSE_NUM_JOINTS:
        print(f"Cảnh báo: num_joints trong cấu hình ({num_joints}) không khớp với số khớp của YOLO-Pose ({YOLO_POSE_NUM_JOINTS}).")
        print(f"Sẽ sử dụng {YOLO_POSE_NUM_JOINTS} khớp của YOLO-Pose.")
        num_joints_to_use = YOLO_POSE_NUM_JOINTS
    else:
        num_joints_to_use = num_joints

    # Kích thước đầu vào mong muốn của YOLO-Pose (ví dụ: 640x640)
    img_size = 640 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chuẩn bị ảnh cho YOLO-Pose: resize, normalize, chuyển sang tensor
        # Tùy thuộc vào implementation của YOLO-Pose, có thể cần các bước khác
        img_h, img_w = frame.shape[:2]
        img_input = cv2.resize(frame, (img_size, img_size))
        img_input = img_input[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img_input = np.ascontiguousarray(img_input)
        img_input = torch.from_numpy(img_input).to(pose_detector_model.device) # Đảm bảo model có thuộc tính device
        img_input = img_input.float() / 255.0  # Normalize to 0-1
        img_input = img_input.unsqueeze(0)  # Add batch dimension

        # Thực hiện inferencing
        # THAY THẾ BẰNG CÁCH GỌI MÔ HÌNH YOLO-POSE CỦA BẠN
        with torch.no_grad():
            results = pose_detector_model(img_input)[0] # Giả định output là tensor đầu tiên

        # Lọc các phát hiện và xử lý NMS (Non-Maximum Suppression)
        # Các bước này có thể đã được xử lý bên trong hàm gọi mô hình YOLO-Pose của bạn
        # Ví dụ: results = non_max_suppression(results, conf_thres=0.25, iou_thres=0.45)
        
        current_frame_keypoints = np.zeros((max_persons, num_joints_to_use, in_channels), dtype=np.float32)
        person_count = 0

        # Lặp qua từng phát hiện trong khung hình
        # Dạng của `results` sẽ phụ thuộc vào cách YOLO-Pose 11m trả về
        # Giả sử `results` là một tensor với mỗi hàng là một phát hiện
        # [x1, y1, x2, y2, conf, class_id, kp1_x, kp1_y, kp1_conf, ..., kp17_x, kp17_y, kp17_conf]
        if results is not None and len(results) > 0:
            for *xyxy, conf, cls, kps in results.cpu().numpy(): # Giải nén bounding box, conf, class, và keypoints
                if int(cls) == 0 and conf > 0.5:  # Chỉ lấy người (class_id=0) và độ tin cậy cao
                    if person_count >= max_persons:
                        break # Đã đạt số lượng người tối đa

                    # Chuyển đổi tọa độ keypoint từ kích thước ảnh đầu vào sang kích thước ảnh gốc
                    # (x, y) keypoint của YOLO-Pose thường là tọa độ chuẩn hóa hoặc tọa độ trên ảnh đầu vào model (640x640)
                    # Bạn cần scale lại về kích thước ảnh gốc (frame.shape)
                    
                    # Giả sử kps là một mảng 1D của (x,y,conf) cho 17 keypoint: [x1,y1,c1, x2,y2,c2, ...]
                    kps = kps.reshape(-1, 3) # Chuyển thành (17, 3)

                    # Scale keypoints trở lại kích thước ảnh gốc
                    # Kps_x = Kps_x_on_resized_img * (original_width / resized_width)
                    # Kps_y = Kps_y_on_resized_img * (original_height / resized_height)
                    kps[:, 0] = kps[:, 0] * (img_w / img_size) # Scale X
                    kps[:, 1] = kps[:, 1] * (img_h / img_size) # Scale Y
                    
                    person_kps = np.array(kps, dtype=np.float32)

                    if in_channels == 2:
                        person_kps = person_kps[:, :2] # Chỉ lấy x, y
                    elif in_channels == 3:
                        # Đảm bảo cột visibility/confidence được đưa vào
                        person_kps = person_kps # Đã có x,y,conf

                    # Đảm bảo số khớp khớp với num_joints_to_use (17)
                    if person_kps.shape[0] == num_joints_to_use:
                        current_frame_keypoints[person_count, :, :] = person_kps
                        person_count += 1
                    else:
                        print(f"Cảnh báo: Số khớp không khớp với {num_joints_to_use} cho một phát hiện.")


        all_frames_keypoints.append(current_frame_keypoints)

    cap.release()

    if not all_frames_keypoints:
        return None
        
    return np.stack(all_frames_keypoints, axis=0) # Shape: (T, max_persons, J, C)


def process_data(args, pose_detector_model): # Truyền model vào process_data
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
            
            # Truyền pose_detector_model vào hàm trích xuất
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

# --- Hàm tải mô hình YOLO-Pose 11m thực tế ---
# Bạn cần thay thế phần này bằng code tải mô hình YOLO-Pose 11m của bạn.
# Dưới đây là một ví dụ về cách bạn có thể cấu trúc nó nếu bạn đang sử dụng một repo YOLOv7-pose
# Có thể bạn sẽ cần clone https://github.com/WongKinYiu/yolov7
# và cài đặt các dependency của nó.
# Sau đó, tải trọng số yolov7-w6-pose.pt hoặc tương tự nếu có yolopose 11m.
def load_yolopose_model(weights_path, config_path=None):
    # Đảm bảo bạn đã cài đặt PyTorch và các dependency của YOLO-Pose
    try:
        import torch
        # from models.yolo import Model # Thay thế bằng cách import thực tế
        # from utils.general import non_max_suppression, scale_coords, check_img_size
        # from utils.torch_utils import select_device

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Sử dụng thiết bị: {device}")

        # --- ĐÂY LÀ PHẦN BẠN CẦN THAY THẾ BẰNG LOGIC TẢI YOLO-POSE 11M THỰC TẾ ---
        # Ví dụ: Nếu bạn đã clone repo YOLOv7 và có file .pt
        # model = Model(cfg=config_path if config_path else 'path/to/yolov7-pose.yaml', ch=3, nc=1).to(device)
        # ckpt = torch.load(weights_path, map_location=device)
        # model.load_state_dict(ckpt['model'].float().state_dict())
        # model.eval()
        # return model, device
        
        # --- MOCK MODEL CHO MỤC ĐÍCH TEST CẤU TRÚC ---
        class MockYOLOPoseModel:
            def __init__(self, device):
                self.device = device
                print(f"Mô hình YOLO-Pose giả lập đã được khởi tạo trên {device}.")
            
            def __call__(self, img_tensor):
                # Giả lập đầu ra của YOLO-Pose:
                # [batch_idx, x1, y1, x2, y2, conf, class_id, kp1_x, kp1_y, kp1_conf, ..., kp17_x, kp17_y, kp17_conf]
                # img_tensor.shape: (1, 3, H, W)
                batch_size, _, img_h, img_w = img_tensor.shape

                all_detections = []
                for b in range(batch_size):
                    num_dummy_persons = 1 # Hoặc np.random.randint(1, args.max_persons + 1)
                    dummy_results_list = []
                    for _ in range(num_dummy_persons):
                        # Bbox: [x1,y1,x2,y2]
                        x1, y1 = np.random.randint(0, img_w // 2), np.random.randint(0, img_h // 2)
                        x2, y2 = np.random.randint(x1 + 50, img_w), np.random.randint(y1 + 50, img_h)
                        bbox = np.array([x1, y1, x2, y2])
                        conf = np.random.rand() * 0.3 + 0.6 # conf ~ [0.6, 0.9]
                        cls_id = 0 # person

                        # Keypoints: 17 * 3 (x,y,conf)
                        kps_data = np.random.rand(17, 3).astype(np.float32)
                        # Scale keypoints relative to bbox for realism
                        kps_data[:, 0] = kps_data[:, 0] * (x2 - x1) + x1
                        kps_data[:, 1] = kps_data[:, 1] * (y2 - y1) + y1
                        kps_data[:, 2] = kps_data[:, 2] * 0.5 + 0.5 # conf ~ [0.5, 1.0]

                        detection = np.concatenate((bbox, [conf, cls_id], kps_data.flatten()))
                        dummy_results_list.append(detection)
                    
                    if len(dummy_results_list) > 0:
                        all_detections.append(torch.tensor(np.array(dummy_results_list), dtype=torch.float32).to(self.device))
                    else:
                        all_detections.append(torch.empty((0, 5 + 17*3), dtype=torch.float32).to(self.device)) # Empty tensor
                
                return all_detections # Trả về list of tensors, mỗi tensor cho 1 batch item
            
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
    args.max_persons = 2 # YOLO-Pose có thể phát hiện nhiều người
    args.num_joints = 17 # YOLO-Pose (COCO) thường có 17 khớp
    args.in_channels = 3 # x,y,confidence

    # Tải mô hình YOLO-Pose 11m
    # THAY THẾ 'path/to/your/yolopose_11m.pt' BẰNG ĐƯỜNG DẪN THỰC TẾ
    # VÀ 'path/to/your/yolopose_11m.yaml' (nếu có file cấu hình)
    
    # Dùng mô hình giả lập nếu bạn chưa có YOLO-Pose 11m thực tế
    pose_model, device = load_yolopose_model(weights_path='dummy_yolopose_11m.pt', config_path='dummy_config.yaml')

    if pose_model is not None:
        process_data(args, pose_model) # Truyền mô hình đã tải vào hàm process_data
    else:
        print("Không thể tải mô hình YOLO-Pose. Vui lòng kiểm tra đường dẫn và cài đặt.")