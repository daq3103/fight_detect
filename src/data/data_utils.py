# data/data_utils.py
import cv2
import os
import numpy as np

def frames_extraction(video_path, image_height, image_width, sequence_length):
    """
    Extracts a fixed number of frames from a video, resizes, and normalizes them.

    Args:
        video_path (str): Path to the video file.
        image_height (int): Desired height of the frames.
        image_width (int): Desired width of the frames.
        sequence_length (int): Number of frames to extract.

    Returns:
        np.array: A NumPy array of extracted frames (sequence_length, image_height, image_width, 3).
                  Returns an empty list if video cannot be processed or not enough frames.
    """
    frames_list = []

    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_frames_count == 0:
        video_reader.release()
        return []

    skip_frames_window = max(int(video_frames_count / sequence_length), 1)

    for frame_counter in range(sequence_length):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()

        if not success:
            break

        resized_frame = cv2.resize(frame, (image_width, image_height)) # cv2.resize takes (width, height)
        normalized_frame = resized_frame / 255.0  # Normalize to [0, 1]
        frames_list.append(normalized_frame)

    video_reader.release()

    if len(frames_list) == sequence_length:
        return np.array(frames_list)
    else:
        # print(f"Warning: Video {os.path.basename(video_path)} has {len(frames_list)} frames, expected {sequence_length}.")
        return []