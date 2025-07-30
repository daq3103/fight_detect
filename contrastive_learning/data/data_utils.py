# data/data_utils.py
import cv2
import numpy as np


def frames_extraction(video_path, image_height, image_width, sequence_length):
    """
    Trích xuất một số lượng frame cố định từ video, thay đổi kích thước và chuẩn hóa chúng.
    """
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_frames_count < sequence_length:
        video_reader.release()
        return None  # Trả về None nếu video quá ngắn

    skip_frames_window = max(int(video_frames_count / sequence_length), 1)

    for frame_counter in range(sequence_length):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break

        resized_frame = cv2.resize(frame, (image_width, image_height))
        # Chuẩn hóa về [0, 1]
        # normalized_frame = resized_frame / 255.0
        frames_list.append(resized_frame)

    video_reader.release()

    if len(frames_list) == sequence_length:
        return np.array(frames_list, dtype=np.float32)
    else:
        return None