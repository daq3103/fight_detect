import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import time
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# --- Cập nhật imports ---
from configs.configs import parse_arguments 
from models.yolopose_stgcn import OptimizedFightDetector # Sửa 'your_model_file'
from data.datasets import PoseDataset # Sử dụng PoseDataset đã viết ở câu trả lời trước
from utils.callbacks import EarlyStopping
from trainer.trainer import Trainer3DCNN # Giữ nguyên tên, class vẫn tương thích
from utils.viz import plot_combined_metrics 

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_arguments()
    set_seed(args.seed)

    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 2. Model initialization
    model = OptimizedFightDetector(
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        num_joints=args.num_joints,
        max_persons=args.max_persons,
    )
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tổng số tham số có thể huấn luyện: {total_params:,}")

    # 3. Loss function
    criterion = nn.CrossEntropyLoss()

    # 4. Optimizer
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

    # 5. Data preparation (THAY ĐỔI LỚN)
    # Tạo dataset riêng cho train và validation
    
    # Đường dẫn tới các file đã được tiền xử lý
    train_data_path = os.path.join(args.data_preprocessed_dir, "train_data.npy")
    train_labels_path = os.path.join(args.data_preprocessed_dir, "train_labels.pkl")
    val_data_path = os.path.join(args.data_preprocessed_dir, "val_data.npy")
    val_labels_path = os.path.join(args.data_preprocessed_dir, "val_labels.pkl")

    if not (os.path.exists(train_data_path) and os.path.exists(val_data_path)):
        print(f"Lỗi: Không tìm thấy dữ liệu đã tiền xử lý tại '{args.data_preprocessed_dir}'.")
        print("Vui lòng chạy script 'preprocess_poses.py' trước.")
        return

    # Khởi tạo Train Dataset
    train_dataset = PoseDataset(
        data_path=train_data_path,
        label_path=train_labels_path,
        sequence_length=args.sequence_length,
        max_persons=args.max_persons,
        num_joints=args.num_joints,
        in_channels=args.in_channels,
        is_train=True  # Bật chế độ train (augmentation, random sampling)
    )

    # Khởi tạo Validation Dataset
    val_dataset = PoseDataset(
        data_path=val_data_path,
        label_path=val_labels_path,
        sequence_length=args.sequence_length,
        max_persons=args.max_persons,
        num_joints=args.num_joints,
        in_channels=args.in_channels,
        is_train=False # Tắt chế độ train (lấy mẫu ở giữa)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Số mẫu huấn luyện: {len(train_dataset)}")
    print(f"Số mẫu validation: {len(val_dataset)}")

    # 6. Callbacks
    callbacks = []
    os.makedirs(args.model_save_path, exist_ok=True)
    early_stopping_callback = EarlyStopping(
        patience=args.es_patience,
        path=os.path.join(args.model_save_path, 'best_model_es.pth'),
        monitor=args.es_monitor,
    )
    callbacks.append(early_stopping_callback)
    reduce_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min" if "loss" in args.lr_reduce_monitor else "max",
        factor=args.lr_reduce_factor,
        patience=args.lr_reduce_patience,
    )
    callbacks.append(reduce_lr_scheduler)

    # 7. Initialize and run Trainer
    trainer = Trainer3DCNN( # Tên class cũ nhưng vẫn dùng tốt
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        callbacks=callbacks,
        model_save_path=args.model_save_path,
        use_amp=args.use_amp,
        grad_accum_steps=args.grad_accum_steps,
    )

    model_history = trainer.train(num_epochs=args.epochs)

    print("\nLịch sử huấn luyện cuối cùng:")
    print(model_history)

    # 8. Plotting the results
    plot_combined_metrics(model_history, save_path=args.model_save_path)

if __name__ == "__main__":
    main()