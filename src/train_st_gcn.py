import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import os
import time
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from configs.configs import parse_arguments # Giả định file này được cập nhật để có thêm các args mới
# Thay đổi import model
from models.yolopose_stgcn import OptimizedFightDetector # <-- THAY ĐỔI: Thay 'your_model_file' bằng tên file chứa model của bạn

# THAY ĐỔI: Bạn cần tạo một Dataset mới để tải dữ liệu keypoints.
# Dưới đây là một ví dụ về cấu trúc của PoseDataset.
# Bạn cần thay thế nó bằng implementation thực tế của mình.
from data.datasets import PoseDataset # <-- THAY ĐỔI: Sử dụng PoseDataset

from utils.callbacks import EarlyStopping
from trainer.trainer import Trainer3DCNN # Giữ nguyên Trainer, nó vẫn tương thích
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

    # 2. Model initialization (THAY ĐỔI)
    model = OptimizedFightDetector(
        num_classes=args.num_classes,
        in_channels=args.in_channels, # Thường là 2 (x, y) hoặc 3 (x, y, confidence)
        num_joints=args.num_joints,
        max_persons=args.max_persons,
    )
    model.to(device)
    
    # In ra tổng số tham số của model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tổng số tham số có thể huấn luyện: {total_params:,}")


    # 3. Loss function
    criterion = nn.CrossEntropyLoss()

    # 4. Optimizer
    if args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported.")

    # 5. Data preparation (THAY ĐỔI LỚN)
    if not os.path.exists(args.data_preprocessed_dir):
        print(f"Lỗi: Không tìm thấy thư mục dataset tại {args.data_preprocessed_dir}")
        print("Vui lòng đảm bảo dữ liệu keypoints đã được tiền xử lý và lưu đúng chỗ.")
        return

    # Sử dụng PoseDataset thay vì VideoDataset
    full_dataset = PoseDataset(
        data_path=os.path.join(args.data_preprocessed_dir, "train_data.npy"), # Ví dụ đường dẫn
        label_path=os.path.join(args.data_preprocessed_dir, "train_label.pkl"), # Ví dụ đường dẫn
        max_persons=args.max_persons
    )

    # Chia dataset thành train và validation
    train_size = int((1 - args.val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    if train_size == 0 or val_size == 0:
         print(f"Lỗi: Dataset quá nhỏ để chia (tổng cộng {len(full_dataset)} mẫu). Vui lòng cung cấp thêm dữ liệu.")
         return

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"Tổng số mẫu trong dataset: {len(full_dataset)}")
    print(f"Số mẫu huấn luyện: {len(train_dataset)}")
    print(f"Số mẫu validation: {len(val_dataset)}")

    # 6. Callbacks
    callbacks = []

    # Tạo đường dẫn lưu model nếu chưa tồn tại
    os.makedirs(args.model_save_path, exist_ok=True)
    
    early_stopping_callback = EarlyStopping(
        patience=args.es_patience,
        verbose=args.es_verbose,
        delta=args.es_delta,
        path=os.path.join(args.model_save_path, 'best_model_es.pth'), # Sửa đường dẫn để không ghi đè
        monitor=args.es_monitor,
        restore_best_weights=args.es_restore_best_weights,
    )
    callbacks.append(early_stopping_callback)

    reduce_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min" if "loss" in args.lr_reduce_monitor else "max",
        factor=args.lr_reduce_factor,
        patience=args.lr_reduce_patience,
        min_lr=args.lr_reduce_min_lr,
        verbose=args.lr_reduce_verbose,
    )
    callbacks.append(reduce_lr_scheduler)

    # 7. Initialize and run Trainer
    # Lớp Trainer3DCNN vẫn hoạt động tốt vì nó được thiết kế khá chung chung
    trainer = Trainer3DCNN(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        callbacks=callbacks,
        model_save_path=args.model_save_path,
        use_amp=args.use_amp,
        grad_accum_steps=args.grad_accum_steps
    )

    # Đổi tên biến history để phù hợp hơn
    model_history = trainer.train(num_epochs=args.epochs)

    print("\nLịch sử huấn luyện cuối cùng:")
    for key, values in model_history.items():
        print(f"{key}: {values}")

    # 8. Plotting the results
    plot_combined_metrics(model_history, save_path=args.model_save_path)


if __name__ == "__main__":
    main()