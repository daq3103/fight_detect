# preprocess_data.py
import os
import glob
from tqdm import tqdm
import numpy as np
from data_utils import frames_extraction # Tái sử dụng hàm của bạn
from configs.configs import parse_arguments


args = parse_arguments()

SOURCE_DATA_DIR = args.data_raw_dir
# Đường dẫn để lưu các frame đã xử lý
DEST_DATA_DIR = args.data_preprocessed_dir
CLASSES_LIST = args.classes_list
# Các tham số trích xuất frame (giữ nguyên như trong config của bạn)
SEQUENCE_LENGTH = args.sequence_length
IMAGE_HEIGHT = args.image_height
IMAGE_WIDTH = args.image_width

# --- Bắt đầu xử lý ---
if not os.path.exists(DEST_DATA_DIR):
    os.makedirs(DEST_DATA_DIR)

for class_name in CLASSES_LIST:
    print(f"Đang xử lý lớp: {class_name}")
    
    # Tạo thư mục con trong thư mục đích
    dest_class_path = os.path.join(DEST_DATA_DIR, class_name)
    if not os.path.exists(dest_class_path):
        os.makedirs(dest_class_path)
        
    # Lấy tất cả đường dẫn video của lớp hiện tại
    source_class_path = os.path.join(SOURCE_DATA_DIR, class_name)
    video_paths = glob.glob(os.path.join(source_class_path, "*.mp4"))

    # Dùng tqdm để xem tiến trình
    for video_path in tqdm(video_paths):
        # Trích xuất frames
        frames = frames_extraction(video_path, IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH)
        
        # Chỉ lưu nếu trích xuất thành công đủ số frame
        if len(frames) == SEQUENCE_LENGTH:
            # Lấy tên file video (không bao gồm đuôi .mp4) để đặt tên cho file .npy
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            save_path = os.path.join(dest_class_path, f"{video_name}.npy")
            
            # Lưu mảng numpy
            np.save(save_path, frames)

print("Hoàn tất tiền xử lý dữ liệu!")