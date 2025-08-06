# train_stage_2_supervised.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import SupervisedVideoDataset, collate_fn 
from models.model_3dcnn import FightDetector3DCNN
from utils.trainer import SupervisedTrainer 
from torchvision.transforms import v2 as T
import random
from configs.default_config import STAGE2_SUPERVISED_CONFIG as config, DEVICE, CLASSES_LIST, IMAGE_HEIGHT, IMAGE_WIDTH
from configs.default_config import SEQUENCE_LENGTH_S2

# Import các thư viện cần thiết cho việc tiền xử lý
import os
import glob
import numpy as np
from tqdm import tqdm

# Import hàm frames_extraction từ data_utils để sử dụng trong hàm tiền xử lý
from data.data_utils import frames_extraction


def temporal_crop_fn(vid, ratio=0.8):
    T0 = vid.size(0)
    L = max(1, int(T0 * ratio))
    start = random.randint(0, T0 - L)
    return vid[start : start + L]

def pre_process_and_save(data_dir, output_dir, classes_list, sequence_length):
    """
    Hàm tiền xử lý dữ liệu và lưu frames dưới dạng tệp .npy.
    Nếu dữ liệu đã tồn tại, sẽ bỏ qua để tiết kiệm thời gian.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    is_processed_all = True
    
    print(f"--- Kiểm tra và tiền xử lý dữ liệu tại: {data_dir} ---")
    for class_name in tqdm(classes_list, desc="Processing classes"):
        class_path = os.path.join(data_dir, class_name)
        output_class_path = os.path.join(output_dir, class_name)
        
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path, exist_ok=True)
            
        for ext in ("*.avi", "*.mp4"):
            video_paths = glob.glob(os.path.join(class_path, ext))
            for video_path in tqdm(video_paths, desc=f"  Processing videos in '{class_name}'"):
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                output_file_path = os.path.join(output_class_path, f"{video_name}.npy")

                # Kiểm tra xem tệp đã tồn tại chưa để tránh xử lý lại
                if os.path.exists(output_file_path):
                    continue

                is_processed_all = False
                frames = frames_extraction(video_path, IMAGE_HEIGHT, IMAGE_WIDTH, sequence_length)

                if frames is not None:
                    # Lưu frames dưới dạng mảng NumPy
                    np.save(output_file_path, frames)
    
    if not is_processed_all:
        print(f"Tiền xử lý hoàn tất cho thư mục: {output_dir}")
    else:
        print(f"Dữ liệu tại {output_dir} đã được tiền xử lý trước đó.")
    
    return not is_processed_all # Trả về True nếu có tệp mới được xử lý

def main():
    print("--- Bắt đầu Giai đoạn 2: Supervised Fine-tuning ---")

    # Xác định đường dẫn cho dữ liệu đã được tiền xử lý
    processed_data_path = f"{config['data_path']}_processed"
    train_processed_path = os.path.join(processed_data_path, 'train')
    val_processed_path = os.path.join(processed_data_path, 'val')

    # Bước 1: Tiền xử lý dữ liệu trước khi huấn luyện
    # Lần chạy đầu tiên sẽ tạo tệp .npy, các lần sau sẽ bỏ qua
    pre_process_and_save(
        data_dir=os.path.join(config['data_path'], 'train'),
        output_dir=train_processed_path,
        classes_list=CLASSES_LIST,
        sequence_length=SEQUENCE_LENGTH_S2
    )
    pre_process_and_save(
        data_dir=os.path.join(config['data_path'], 'val'),
        output_dir=val_processed_path,
        classes_list=CLASSES_LIST,
        sequence_length=SEQUENCE_LENGTH_S2
    )

    # 1. DataLoader
    # SỬ DỤNG ĐƯỜNG DẪN DỮ LIỆU ĐÃ ĐƯỢC TIỀN XỬ LÝ
    train_dataset = SupervisedVideoDataset(
        data_dir=train_processed_path,
        classes_list=CLASSES_LIST,
        sequence_length=SEQUENCE_LENGTH_S2,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        # transform=transform
    )

    val_dataset = SupervisedVideoDataset(
        data_dir=val_processed_path,
        classes_list=CLASSES_LIST,
        sequence_length=SEQUENCE_LENGTH_S2,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    print(f"Đã tải {len(train_dataset)} mẫu train và {len(val_dataset)} mẫu val.")

    # 2. Model
    model = FightDetector3DCNN(num_classes=len(CLASSES_LIST))
    print(f"Tải trọng số từ Giai đoạn 1: {config['stage1_best_model_path']}")
    model.load_state_dict(torch.load(config['stage1_best_model_path'], map_location=DEVICE), strict=False)
    
    # Chuẩn bị model cho Giai đoạn 2: Đóng băng backbone
    model.prepare_for_finetuning_classifier()

    # 3. Optimizer, Loss, Scheduler
    # Chỉ train các tham số của classifier_head
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(params, lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3, verbose=True)

    # 4. Trainer
    trainer = SupervisedTrainer( # SupervisedTrainer vẫn dùng lại được
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        lr_scheduler=scheduler,
        config=config,
    )
    
    trainer.train()

if __name__ == "__main__":
    main()