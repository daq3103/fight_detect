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
    
    # Override paths based on environment
    import sys
    if 'kaggle' in sys.path[0].lower() or '/kaggle/' in os.getcwd():
        # Running on Kaggle
        if not hasattr(args, 'data_raw_dir') or args.data_raw_dir.startswith('/kaggle/input/violencedataset/'):
            args.data_raw_dir = "/kaggle/input/fight-data"  # Kaggle input path
            print(f"Kaggle environment detected. Using path: {args.data_raw_dir}")
        
        if not hasattr(args, 'data_preprocessed_dir') or args.data_preprocessed_dir.startswith('/kaggle/working/'):
            args.data_preprocessed_dir = "/kaggle/working/preprocessed_data"  # Kaggle output path
            print(f"Output will be saved to: {args.data_preprocessed_dir}")
    else:
        # Running locally
        if args.data_raw_dir.startswith('/kaggle/'):
            args.data_raw_dir = r"D:\code\FightDetection\dataset"
            print(f"Local environment detected. Using path: {args.data_raw_dir}")
        
        if args.data_preprocessed_dir.startswith('/kaggle/'):
            args.data_preprocessed_dir = r"D:\code\FightDetection\data\preprocessed"
            print(f"Output will be saved to: {args.data_preprocessed_dir}")

    # Check if the input directory exists
    if not os.path.exists(args.data_raw_dir):
        print(f"Lỗi: Thư mục dữ liệu gốc không tồn tại: {args.data_raw_dir}")
        print("Vui lòng kiểm tra lại đường dẫn dữ liệu.")
        print("Trên Kaggle, đảm bảo dataset đã được add vào notebook.")
        return

    # List contents of data directory for debugging
    print(f"\nNội dung của thư mục {args.data_raw_dir}:")
    try:
        for item in os.listdir(args.data_raw_dir):
            item_path = os.path.join(args.data_raw_dir, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
            else:
                print(f"  📄 {item}")
    except Exception as e:
        print(f"Không thể liệt kê nội dung: {e}")

    # In ra các tham số sẽ được sử dụng
    print("\nBắt đầu quá trình tiền xử lý với các tham số sau:")
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