# preprocess_data.py

import os
import glob
import numpy as np
from tqdm import tqdm  # Thư viện để hiển thị thanh tiến trình, rất hữu ích!
import argparse

# Import hàm trích xuất frame từ file data_utils.py của bạn
from data.data_utils import frames_extraction
from configs.configs import parse_arguments # Giả sử các config của bạn cũng ở đây

def preprocess_videos(data_dir, output_dir, image_height, image_width, sequence_length, video_extensions=['*.mp4', '*.avi', '*.mov']):
    """
    Duyệt qua thư mục dữ liệu video gốc, trích xuất các khung hình cho mỗi video,
    và lưu chúng dưới dạng tệp .npy trong thư mục đầu ra.

    Args:
        data_dir (str): Đường dẫn đến thư mục chứa các video gốc (ví dụ: 'Real Life Violence Dataset').
                        Thư mục này nên chứa các thư mục con cho mỗi lớp (ví dụ: 'Fight', 'NoFight').
        output_dir (str): Đường dẫn đến thư mục để lưu các tệp .npy đã xử lý.
        image_height (int): Chiều cao mong muốn của khung hình.
        image_width (int): Chiều rộng mong muốn của khung hình.
        sequence_length (int): Số lượng khung hình cần trích xuất từ mỗi video.
        video_extensions (list): Danh sách các phần mở rộng của file video cần tìm.
    """
    # Lấy danh sách các lớp từ tên các thư mục con trong data_dir
    classes_list = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not classes_list:
        print(f"Lỗi: Không tìm thấy thư mục lớp nào trong '{data_dir}'.")
        return

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

    # In ra các tham số sẽ được sử dụng
    print("Bắt đầu quá trình tiền xử lý với các tham số sau:")
    print(f"Thư mục video gốc: {args.data_dir}")
    print(f"Thư mục đầu ra: {args.data_preprocessed_dir}")
    print(f"Kích thước ảnh (H x W): {args.image_height} x {args.image_width}")
    print(f"Số lượng frames mỗi video: {args.sequence_length}")

    # Gọi hàm tiền xử lý
    preprocess_videos(
        data_dir=args.data_dir,
        output_dir=args.data_preprocessed_dir,
        image_height=args.image_height,
        image_width=args.image_width,
        sequence_length=args.sequence_length
    )


if __name__ == "__main__":
    main()