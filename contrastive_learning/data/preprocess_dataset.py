# pre_process_dataset.py
import os
import glob
import numpy as np
from tqdm import tqdm
from data.data_utils import frames_extraction
from configs.default_config import CLASSES_LIST, IMAGE_HEIGHT, IMAGE_WIDTH
import numpy as np

def pre_process_and_save(data_dir, output_dir, classes_list, sequence_length):
    """
    Quét qua các video, trích xuất frames và lưu dưới dạng tệp .npy
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_name in tqdm(classes_list, desc="Processing classes"):
        class_path = os.path.join(data_dir, class_name)
        output_class_path = os.path.join(output_dir, class_name)
        
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)
            
        for ext in ("*.avi", "*.mp4"):
            video_paths = glob.glob(os.path.join(class_path, ext))
            for video_path in tqdm(video_paths, desc=f"  Processing videos in '{class_name}'"):
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                output_file_path = os.path.join(output_class_path, f"{video_name}.npy")

                # Kiểm tra xem tệp đã tồn tại chưa để tránh xử lý lại
                if os.path.exists(output_file_path):
                    continue

                # Trích xuất frames (giữ nguyên logic từ code của bạn)
                frames = frames_extraction(video_path, IMAGE_HEIGHT, IMAGE_WIDTH, sequence_length)

                if frames is not None:
                    # Lưu frames dưới dạng mảng NumPy
                    np.save(output_file_path, frames)

def main():
    # Sử dụng cùng các biến config để đảm bảo đồng nhất
    from configs.default_config import STAGE2_SUPERVISED_CONFIG as config
    from configs.default_config import SEQUENCE_LENGTH_S2
    
    # Định nghĩa thư mục lưu trữ frames đã được xử lý
    train_output_path = f"{config['data_path']}_processed/train"
    val_output_path = f"{config['data_path']}_processed/val"

    print("--- Bắt đầu tiền xử lý dữ liệu train ---")
    pre_process_and_save(
        data_dir=f"{config['data_path']}/train",
        output_dir=train_output_path,
        classes_list=CLASSES_LIST,
        sequence_length=SEQUENCE_LENGTH_S2
    )

    print("--- Bắt đầu tiền xử lý dữ liệu val ---")
    pre_process_and_save(
        data_dir=f"{config['data_path']}/val",
        output_dir=val_output_path,
        classes_list=CLASSES_LIST,
        sequence_length=SEQUENCE_LENGTH_S2
    )
    
    print("--- Tiền xử lý hoàn tất! ---")

if __name__ == "__main__":
    main()