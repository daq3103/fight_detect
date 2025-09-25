# preprocess_data.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import glob
from data.data_utils import frames_extraction
from configs.configs import parse_arguments

import numpy as np
from tqdm import tqdm  # Thư viện để hiển thị thanh tiến trình, rất hữu ích!
import argparse

def preprocess_videos(data_dir, output_dir, image_height, image_width, sequence_length, video_extensions=['*.mp4', '*.avi', '*.mov']):

    classes_list = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]


    print(f"Tìm thấy các lớp: {classes_list}")

    # Tạo thư mục đầu ra nếu nó chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Lặp qua từng lớp
    for class_name in classes_list:
        print(f"\nĐang xử lý lớp: {class_name}")
        
        # Tạo thư mục con cho lớp trong thư mục đầu ra
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # Lấy tất cả các đường dẫn video cho lớp hiện tại
        video_paths = []
        class_input_dir = os.path.join(data_dir, class_name)
        for ext in video_extensions:
            video_paths.extend(glob.glob(os.path.join(class_input_dir, ext)))

        if not video_paths:
            print(f"Cảnh báo: Không tìm thấy video nào cho lớp '{class_name}'.")
            continue

        # Lặp qua từng video trong lớp và xử lý nó
        for video_path in tqdm(video_paths, desc=f"Trích xuất frames cho '{class_name}'"):
            # Trích xuất frames sử dụng hàm bạn đã cung cấp
            frames = frames_extraction(video_path, image_height, image_width, sequence_length)

            # Chỉ lưu nếu trích xuất thành công (trả về một mảng không rỗng)
            if frames is not None and len(frames) == sequence_length:
                # Tạo đường dẫn file .npy đầu ra
                video_filename = os.path.basename(video_path)
                video_name_without_ext = os.path.splitext(video_filename)[0]
                output_npy_path = os.path.join(class_output_dir, f"{video_name_without_ext}.npy")

                # Lưu mảng frames vào file .npy
                np.save(output_npy_path, frames)
            else:
                print(f"Cảnh báo: Bỏ qua video '{video_path}' do không đủ số lượng frames hoặc lỗi đọc file.")

    print("\nHoàn tất quá trình tiền xử lý dữ liệu!")
    print(f"Tất cả các tệp .npy đã được lưu vào: {output_dir}")


def main():
    """
    Hàm chính để chạy script từ dòng lệnh.
    """
    # Sử dụng lại trình phân tích đối số từ file config của bạn
    args = parse_arguments()
    
    # Override paths for local Windows environment if they're still set to Kaggle paths
    if args.data_raw_dir.startswith('/kaggle/'):
        # Update this path to your actual local data directory
        args.data_raw_dir = r"D:\code\FightDetection\dataset"
        print(f"Đã thay đổi đường dẫn từ Kaggle sang local: {args.data_raw_dir}")
    
    if args.data_preprocessed_dir.startswith('/kaggle/'):
        # Update this path to your desired output directory
        args.data_preprocessed_dir = r"D:\code\FightDetection\data\preprocessed"
        print(f"Đã thay đổi đường dẫn output từ Kaggle sang local: {args.data_preprocessed_dir}")

    # Check if the input directory exists
    if not os.path.exists(args.data_raw_dir):
        print(f"Lỗi: Thư mục dữ liệu gốc không tồn tại: {args.data_raw_dir}")
        print("Vui lòng tạo thư mục và đặt dữ liệu video của bạn vào đó.")
        return

    # In ra các tham số sẽ được sử dụng
    print("Bắt đầu quá trình tiền xử lý với các tham số sau:")
    print(f"Thư mục video gốc: {args.data_raw_dir}")
    print(f"Thư mục đầu ra: {args.data_preprocessed_dir}")
    print(f"Kích thước ảnh (H x W): {args.image_height} x {args.image_width}")
    print(f"Số lượng frames mỗi video: {args.sequence_length}")

    # Gọi hàm tiền xử lý
    preprocess_videos(
        data_dir=args.data_raw_dir,
        output_dir=args.data_preprocessed_dir,
        image_height=args.image_height,
        image_width=args.image_width,
        sequence_length=args.sequence_length
    )


if __name__ == "__main__":
    main()