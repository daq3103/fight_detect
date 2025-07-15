# preprocess_poses.py
import os
import glob
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from configs.configs import parse_arguments

def extract_and_pad_poses(results, max_persons, num_joints):
    """
    Trích xuất keypoints từ kết quả của YOLO, chuẩn hóa số người và trả về tensor.
    """
    keypoints_data = results[0].keypoints
    
    # Lấy tọa độ xy và điểm tin cậy (confidence)
    xy = keypoints_data.xy.cpu()  # [num_persons, 17, 2]
    conf = keypoints_data.conf.cpu() # [num_persons, 17]
    
    # Kết hợp thành tensor [num_persons, 17, 3]
    if xy.numel() == 0: # Không phát hiện người nào
        return torch.zeros((max_persons, num_joints, 3))
        
    kpts_with_conf = torch.cat((xy, conf.unsqueeze(-1)), dim=-1)
    
    num_detected_persons = kpts_with_conf.shape[0]
    
    # Tạo một tensor rỗng để chứa kết quả đã padding
    padded_kpts = torch.zeros((max_persons, num_joints, 3), dtype=torch.float32)

    if num_detected_persons > 0:
        # Nếu phát hiện nhiều hơn max_persons, chỉ giữ lại những người có confidence cao nhất
        if num_detected_persons > max_persons:
            # Tính trung bình confidence của mỗi người
            person_scores = torch.mean(conf, dim=1)
            # Lấy chỉ số của top `max_persons` người
            top_indices = torch.topk(person_scores, k=max_persons).indices
            kpts_to_keep = kpts_with_conf[top_indices]
        else:
            kpts_to_keep = kpts_with_conf
        
        # Điền dữ liệu vào tensor đã padding
        num_to_fill = kpts_to_keep.shape[0]
        padded_kpts[:num_to_fill, :, :] = kpts_to_keep

    return padded_kpts

def process_video(video_path, model, args):
    """
    Xử lý một video: trích xuất, lấy mẫu và chuẩn hóa keypoints.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return None

    video_keypoints = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Chuẩn hóa kích thước frame để tăng tốc độ xử lý (tùy chọn nhưng khuyến khích)
        frame_resized = cv2.resize(frame, (args.image_width, args.image_height))
        
        # Chạy YOLO-Pose
        results = model(frame_resized, verbose=False)

        # Trích xuất và pad poses
        padded_poses = extract_and_pad_poses(results, args.max_persons, args.num_joints)
        
        # Chuẩn hóa tọa độ về khoảng [0, 1] theo kích thước frame
        padded_poses[:, :, 0] /= args.image_width
        padded_poses[:, :, 1] /= args.image_height
        
        video_keypoints.append(padded_poses)

    cap.release()
    
    if not video_keypoints:
        print(f"Cảnh báo: Không có frame nào được xử lý cho video {video_path}")
        return None

    # Chuyển list các tensor thành một tensor lớn
    video_array = torch.stack(video_keypoints).numpy() # Shape: (T, P, J, C)
    
    total_frames = len(video_array)
    
    # Bỏ qua video quá ngắn
    if total_frames < args.sequence_length:
        print(f"Cảnh báo: Video {video_path} quá ngắn ({total_frames} frames), bỏ qua.")
        return None
        
    # Lấy mẫu đều các frame để có độ dài sequence_length
    indices = np.linspace(0, total_frames - 1, args.sequence_length, dtype=int)
    sampled_frames = video_array[indices]
    
    return sampled_frames


def main():
    args = parse_arguments()
    
    # 1. Tải mô hình YOLOv8-Pose
    print(f"Đang tải mô hình {args.yolo_model_name}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo_model = YOLO(args.yolo_model).to(device)
    print(f"Mô hình đã được tải lên {device}.")

    # 2. Tạo thư mục đích nếu chưa có
    if not os.path.exists(args.data_preprocessed_dir):
        os.makedirs(args.data_preprocessed_dir)
        print(f"Đã tạo thư mục: {args.data_preprocessed_dir}")

    # 3. Lặp qua từng lớp và xử lý video
    for class_name in args.classes_list:
        print(f"\n--- Bắt đầu xử lý lớp: {class_name} ---")
        
        source_class_path = os.path.join(args.data_raw_dir, class_name)
        dest_class_path = os.path.join(args.data_preprocessed_dir, class_name)
        if not os.path.exists(dest_class_path):
            os.makedirs(dest_class_path)

        video_paths = glob.glob(os.path.join(source_class_path, "*.mp4")) + \
                      glob.glob(os.path.join(source_class_path, "*.avi")) # Thêm các định dạng khác nếu cần
        
        if not video_paths:
            print(f"Không tìm thấy video nào trong: {source_class_path}")
            continue

        for video_path in tqdm(video_paths, desc=f"Xử lý {class_name}"):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            save_path = os.path.join(dest_class_path, f"{video_name}.npy")
            
            # Bỏ qua nếu đã xử lý rồi
            if os.path.exists(save_path):
                continue
                
            # Xử lý video
            keypoints_data = process_video(video_path, yolo_model, args)
            
            # Lưu lại nếu xử lý thành công
            if keypoints_data is not None:
                np.save(save_path, keypoints_data)
    
    print("\nHoàn tất tiền xử lý dữ liệu!")

if __name__ == '__main__':
    main()