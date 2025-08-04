import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
import argparse
from tqdm import tqdm
from gradcam.grad_cam_3d import GradCAM3D

# Import model của cậu
from models.model_3dcnn import FightDetector3DCNN

from configs.default_config import (
    DEVICE,
    SEQUENCE_LENGTH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)

CLASSES_LIST = ["Fight", "NonFight"]


class InferenceEngine:
    def __init__(self, model_path, target_layer_name="backbone.layer4"):
        print(f"Loading model from: {model_path}")
        self.model = FightDetector3DCNN(num_classes=len(CLASSES_LIST)).to(DEVICE)

        # Load state dict, bỏ qua các key không khớp (an toàn hơn)
        state_dict = torch.load(model_path, map_location=DEVICE)
        # Xử lý prefix 'module.' nếu model được train bằng DataParallel
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        print("Model loaded successfully.")

        # Khởi tạo GradCAM
        try:
            target_layer = self.model.get_submodule(target_layer_name)
            self.grad_cam = GradCAM3D(self.model, target_layer)
            print(f"GradCAM initialized on layer: {target_layer_name}")
        except AttributeError:
            print(f"Error: Could not find layer '{target_layer_name}'.")
            self.grad_cam = None

        # Chuẩn bị transform cho video frame
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
                ),
            ]
        )

    def _preprocess_clip(self, clip_frames):
        """Chuyển list các frame thành tensor chuẩn cho model"""
        processed_frames = [self.transform(frame) for frame in clip_frames]
        # Stack và đổi chiều [T, C, H, W] -> [C, T, H, W]
        clip_tensor = torch.stack(processed_frames).permute(1, 0, 2, 3)
        # Thêm chiều batch [1, C, T, H, W]
        return clip_tensor.unsqueeze(0)

    def _postprocess_heatmap(
        self, heatmap, frame_shape, heatmap_threshold=0.6, min_area_ratio=0.01
    ):
        """Từ heatmap -> bounding boxes"""
        # Upscale heatmap về kích thước frame gốc
        heatmap_resized = cv2.resize(heatmap, (frame_shape[1], frame_shape[0]))

        # Threshold để lấy vùng "nóng" nhất
        _, thresh = cv2.threshold(
            heatmap_resized, heatmap_threshold, 1, cv2.THRESH_BINARY
        )
        thresh = thresh.astype(np.uint8)

        # Tìm contours của các vùng nóng
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []
        min_area = frame_shape[0] * frame_shape[1] * min_area_ratio
        for cnt in contours:
            # Lọc các box quá nhỏ (nhiễu)
            if cv2.contourArea(cnt) > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append((x, y, w, h))
        return boxes

    def process_video(self, input_path, output_path, detection_threshold=0.8):
        if not self.grad_cam:
            print("GradCAM not available. Exiting.")
            return

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return

        # Lấy thông số video để ghi file output
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_buffer = []

        with tqdm(total=total_frames, desc="Processing Video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR (OpenCV) sang RGB (PyTorch)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_buffer.append(frame_rgb)

                if len(frames_buffer) == SEQUENCE_LENGTH:
                    # 1. Preprocess clip
                    input_tensor = self._preprocess_clip(frames_buffer)

                    # 2. Model prediction
                    with torch.no_grad():
                        logits = self.model(input_tensor.to(DEVICE))
                        probs = F.softmax(logits, dim=1)
                        confidence, pred_idx = torch.max(probs, dim=1)

                    pred_class = CLASSES_LIST[pred_idx.item()]
                    confidence = confidence.item()

                    # 3. Grad-CAM và vẽ box NẾU là 'Fight'
                    if pred_class == "Fight" and confidence >= detection_threshold:
                        fight_class_idx = CLASSES_LIST.index("Fight")
                        heatmaps_3d = self.grad_cam(input_tensor, fight_class_idx)

                        # Xử lý từng frame trong buffer
                        for i in range(SEQUENCE_LENGTH):
                            original_frame = cv2.cvtColor(
                                frames_buffer[i], cv2.COLOR_RGB2BGR
                            )
                            heatmap_2d = heatmaps_3d[i]
                            boxes = self._postprocess_heatmap(
                                heatmap_2d, original_frame.shape
                            )

                            for x, y, w, h in boxes:
                                cv2.rectangle(
                                    original_frame,
                                    (x, y),
                                    (x + w, y + h),
                                    (0, 0, 255),
                                    2,
                                )
                                label = f"{pred_class}: {confidence:.2f}"
                                cv2.putText(
                                    original_frame,
                                    label,
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 0, 255),
                                    2,
                                )
                            out.write(original_frame)
                    else:
                        # Nếu không phải Fight, ghi frame gốc
                        for i in range(SEQUENCE_LENGTH):
                            original_frame = cv2.cvtColor(
                                frames_buffer[i], cv2.COLOR_RGB2BGR
                            )
                            out.write(original_frame)

                    # Xóa frame đầu tiên để nhận frame mới (sliding window)
                    frames_buffer.pop(0)

                pbar.update(1)

        # Xử lý các frame còn lại trong buffer
        for frame_rgb in frames_buffer:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processing complete. Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fight Detection Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained Stage 2 model.",
    )
    parser.add_argument(
        "--input_video", type=str, required=True, help="Path to the input video."
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default="output.mp4",
        help="Path to save the output video.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Confidence threshold for detection.",
    )

    args = parser.parse_args()

    engine = InferenceEngine(model_path=args.model_path)
    engine.process_video(
        input_path=args.input_video,
        output_path=args.output_video,
        detection_threshold=args.threshold,
    )
