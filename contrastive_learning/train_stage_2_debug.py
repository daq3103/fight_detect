# train_stage_2_supervised.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import SupervisedVideoDataset, collate_fn
from models.model_3dcnn import FightDetector3DCNN
from utils.trainer import SupervisedTrainer
from utils.callbacks import EarlyStopping
from torchvision.transforms import v2 as T
import random
from configs.default_config import (
    STAGE2_SUPERVISED_CONFIG as config,
    DEVICE,
    CLASSES_LIST,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)
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
            for video_path in tqdm(
                video_paths, desc=f"  Processing videos in '{class_name}'"
            ):
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                output_file_path = os.path.join(output_class_path, f"{video_name}.npy")

                # Kiểm tra xem tệp đã tồn tại chưa để tránh xử lý lại
                if os.path.exists(output_file_path):
                    continue

                is_processed_all = False
                frames = frames_extraction(
                    video_path, IMAGE_HEIGHT, IMAGE_WIDTH, sequence_length
                )

                if frames is not None:
                    # Lưu frames dưới dạng mảng NumPy
                    np.save(output_file_path, frames)

    if not is_processed_all:
        print(f"Tiền xử lý hoàn tất cho thư mục: {output_dir}")
    else:
        print(f"Dữ liệu tại {output_dir} đã được tiền xử lý trước đó.")

    return not is_processed_all  # Trả về True nếu có tệp mới được xử lý


# train_stage_2_supervised_fixed.py

def debug_model_outputs(model, train_loader, device):
    """Debug function để kiểm tra model outputs"""
    model.eval()
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            print(f"Input shape: {inputs.shape}")
            print(f"Output shape: {outputs.shape}")
            print(f"Output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
            print(f"Output mean: {outputs.mean().item():.4f}")
            print(f"Output std: {outputs.std().item():.4f}")
            print(f"Contains NaN: {torch.isnan(outputs).any()}")
            print(f"Contains Inf: {torch.isinf(outputs).any()}")
            break
    model.train()

# def main():
#     print("--- Bắt đầu Giai đoạn 2: Supervised Fine-tuning (FIXED) ---")

#     # ... existing preprocessing code ...

#     # 0. Augmentation - GIẢM AUGMENTATION để debug

 
#     print(f"Đã tải {len(train_dataset)} mẫu train và {len(val_dataset)} mẫu val.")

#     # 2. Model - KHỞI TẠO LẠI TỪ SCRATCH
#     model = FightDetector3DCNN(
#         num_classes=len(CLASSES_LIST),
#         dropout_prob=0.3,  # Giảm dropout xuống
#     )
    
#     # KIỂM TRA VIỆC LOAD WEIGHTS
#     try:
#         print(f"Đang tải trọng số từ: {config['stage1_best_model_path']}")
#         checkpoint = torch.load(config["stage1_best_model_path"], map_location=DEVICE)
        
#         # In ra keys để debug
#         print("Keys in checkpoint:")
#         for key in list(checkpoint.keys())[:10]:  # In 10 keys đầu
#             print(f"  {key}: {checkpoint[key].shape if hasattr(checkpoint[key], 'shape') else type(checkpoint[key])}")
        
#         # Load với strict=False và bắt lỗi
#         missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
#         print(f"Missing keys: {len(missing_keys)}")
#         print(f"Unexpected keys: {len(unexpected_keys)}")
        
#         if missing_keys:
#             print("Missing keys:", missing_keys[:5])  # In 5 keys đầu
#         if unexpected_keys:
#             print("Unexpected keys:", unexpected_keys[:5])
            
#     except Exception as e:
#         print(f"❌ Lỗi khi load pretrained weights: {e}")
#         print("🔄 Khởi tạo model từ đầu...")
#         # Khởi tạo lại model với pretrained ImageNet
#         model.backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
#         model.backbone.fc = nn.Identity()

#     if torch.cuda.device_count() > 1:
#         print(f"Sử dụng {torch.cuda.device_count()} GPU!")
#         model = nn.DataParallel(model)

#     # DEBUG MODEL OUTPUTS
#     print("\n=== DEBUGGING MODEL OUTPUTS ===")
#     debug_model_outputs(model, train_loader, DEVICE)
    
#     # 3. OPTIMIZER VỚI LR CAO HỞN VÀ SCHEDULER ĐƠN GIẢN
#     optimizer = optim.AdamW(
#         model.parameters(), 
#         lr=1e-4,  # LR cao hơn nhiều
#         weight_decay=1e-4
#     )

def main():
    print("--- Bắt đầu Giai đoạn 2: Supervised Fine-tuning ---")

    # Xác định đường dẫn cho dữ liệu đã được tiền xử lý.
    # Thay đổi đường dẫn lưu trữ để trỏ đến thư mục làm việc của Kaggle.
    # Thư mục /kaggle/working có quyền ghi.
    processed_data_path = f"/kaggle/working/data_processed"
    train_processed_path = os.path.join(processed_data_path, "train")
    val_processed_path = os.path.join(processed_data_path, "val")

    # Bước 1: Tiền xử lý dữ liệu trước khi huấn luyện
    # ... (phần code này không cần thay đổi) ...
    pre_process_and_save(
        data_dir=os.path.join(config["data_path"], "train"),
        output_dir=train_processed_path,
        classes_list=CLASSES_LIST,
        sequence_length=SEQUENCE_LENGTH_S2,
    )
    pre_process_and_save(
        data_dir=os.path.join(config["data_path"], "val"),
        output_dir=val_processed_path,
        classes_list=CLASSES_LIST,
        sequence_length=SEQUENCE_LENGTH_S2,
    )
    # 0. Augmentation
    transform = T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(
            mean=[0.43216, 0.394666, 0.37645], 
            std=[0.22803, 0.22145, 0.216989]
        ),
        # Tạm thời tắt augmentation để debug
        # T.RandomResizedCrop(...),
        # T.RandomHorizontalFlip(...),
    ])

    val_transform = T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(
            mean=[0.43216, 0.394666, 0.37645], 
            std=[0.22803, 0.22145, 0.216989]
        ),
    ])


    # 1. DataLoader
    # Vẫn sử dụng đường dẫn dữ liệu đã tiền xử lý
    train_dataset = SupervisedVideoDataset(
        data_dir=train_processed_path,
        classes_list=CLASSES_LIST,
        sequence_length=SEQUENCE_LENGTH_S2,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        transform=transform,
    )

    val_dataset = SupervisedVideoDataset(
        data_dir=val_processed_path,
        classes_list=CLASSES_LIST,
        sequence_length=SEQUENCE_LENGTH_S2,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        # collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    print(f"Đã tải {len(train_dataset)} mẫu train và {len(val_dataset)} mẫu val.")

    # 2. Model
    model = FightDetector3DCNN(
        num_classes=len(CLASSES_LIST),
        dropout_prob=0.3,  # Giảm dropout xuống
    )
    

    if torch.cuda.device_count() > 1:
        print(f"Sử dụng {torch.cuda.device_count()} GPU!")
        model = nn.DataParallel(model, device_ids=[0, 1])

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=1e-4,  # LR cao hơn nhiều
        weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", factor=0.5, patience=3, verbose=True
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the restart interval
        eta_min=1e-7
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # 5. Callbacks với EARLY STOPPING
    # early_stopping = EarlyStopping(
    #     patience=config['es_patience'],
    #     verbose=True,
    #     delta=0,
    #     path=config['model_save_path'],
    #     monitor='val_acc',
    #     mode='max'
    # )
    early_stopping = EarlyStopping(
        patience=config['es_patience'],
        verbose=True,
        delta=0.001,
        path=config['model_save_path'],
        monitor="val_accuracy",
        restore_best_weights=True,
    )
    # 4. Trainer
    trainer = SupervisedTrainer(  # SupervisedTrainer vẫn dùng lại được
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        lr_scheduler=scheduler,
        early_stopping=early_stopping,
        config=config,
    )

    trainer.train()


if __name__ == "__main__":
    main()
