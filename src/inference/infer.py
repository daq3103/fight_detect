import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from collections import deque
import os
import sys

# Sửa đường dẫn import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import từ thư mục models
from models.model import FightDetectionModel

def main():
    # ============= CẤU HÌNH CỐ ĐỊNH =============
    # Đường dẫn video đầu vào
    input_video = os.path.join(parent_dir, "testing", "input", "V_115.mp4")
    
    # Đường dẫn video đầu ra
    output_video = os.path.join(parent_dir, "testing", "output", "V_115_output.mp4")
    
    # Đường dẫn trọng số mô hình
    weights_path = os.path.join(parent_dir, "weights", "best_mobibilstm_model.pt")
    
    # Tham số mô hình
    img_size = 64  # Phải khớp với kích thước training (64x64)
    seq_len = 64   # Độ dài chuỗi frame
    num_classes = 2
    hidden_size = 32
    dropout_prob = 0.25
    
    # Danh sách lớp
    classes_list = ["Violence", "NonViolence"]
    colors_list = [(0, 0, 255), (0, 255, 0)]  # Xanh: Bình thường, Đỏ: Đánh nhau
    # =============================================

    # Kiểm tra file tồn tại
    if not os.path.exists(input_video):
        print(f"Error: Input video not found at {input_video}")
        return
    
    # Tạo thư mục đầu ra nếu cần
    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Processing video: {input_video}")
    print(f"Output will be saved to: {output_video}")

    # Khởi tạo mô hình
    model = FightDetectionModel(
        num_classes=num_classes,
        hidden_size=hidden_size,
        dropout_prob=dropout_prob,
        image_height=img_size,
        image_width=img_size,
    )
    
    # Tải trọng số
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Error: Weight file not found at {weights_path}")
        return
    
    model.to(device)
    model.eval()

    # Biến đổi hình ảnh
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    frames_queue = deque(maxlen=seq_len)
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video}")
        return

    # Thiết lập đầu ra video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    print("Processing video...")
    prediction_idx = 0  # Dự đoán mặc định: Bình thường (index 0)
    last_confidences = deque(maxlen=5)  # Lưu trữ các confidence gần nhất
    avg_confidence = 0.0  # Giá trị mặc định
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Xử lý frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        processed_frame = transform(pil_image)
        frames_queue.append(processed_frame)

        # Dự đoán khi có đủ chuỗi frame
        if len(frames_queue) == seq_len:
            input_tensor = torch.stack(list(frames_queue), dim=0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)
                prediction_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction_idx].item()
                last_confidences.append(confidence)
                
                current_prediction = classes_list[prediction_idx]
                # Tính confidence trung bình để ổn định kết quả
                if last_confidences:
                    avg_confidence = sum(last_confidences) / len(last_confidences)
                print(f"Prediction: {current_prediction}, Confidence: {avg_confidence:.4f}")

        # Hiển thị kết quả lên frame
        current_prediction_text = classes_list[prediction_idx]
        color = colors_list[prediction_idx]
        
        # Vẽ kết quả với background rõ ràng
        cv2.rectangle(frame, (10, 10), (400, 50), (0, 0, 0), -1)
        status_text = f"Status: {current_prediction_text} ({avg_confidence:.2f})"
        cv2.putText(frame, status_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        # Ghi frame đầu ra
        out.write(frame)
        
        # Hiển thị xem trực tiếp
        cv2.imshow('Fight Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Finished processing. Output saved to {output_video}")


if __name__ == '__main__':
    # Chạy chương trình chính không cần tham số
    main()