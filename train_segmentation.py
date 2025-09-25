# Script để chạy training với dữ liệu đã được tổ chức
import sys
import os
sys.path.append('src')

# Import cần thiết
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import random

from src.configs.configs import parse_arguments
from src.models.model_3dcnn_r2plus1d import FightDetection3DCNN
from src.data.datasets import SegmentationDataset
from src.utils.callbacks import EarlyStopping
from src.trainer.trainer import Trainer3DCNN
from src.utils.viz import plot_combined_metrics

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Tạo args giả lập để test
class Args:
    def __init__(self):
        self.data_preprocessed_dir = "data_organized"
        self.num_classes = 2
        self.sequence_length = 30
        self.image_height = 224
        self.image_width = 224
        self.img_channels = 3
        self.hidden_size = 128
        self.dropout_prob = 0.3
        self.learning_rate = 0.001
        self.optimizer = "AdamW"
        self.batch_size = 8  # Giảm batch size cho test
        self.epochs = 20  # Ít epochs để test
        self.val_split = 0.2
        self.seed = 42
        self.model_save_path = "weights/segmented_model.pth"

        # Learning rate scheduler
        self.lr_reduce_monitor = "val_loss"
        self.lr_reduce_factor = 0.5
        self.lr_reduce_patience = 3
        self.lr_reduce_min_lr = 1e-6

        # Early stopping (disabled for test)
        self.es_patience = 10
        self.es_verbose = True
        self.es_delta = 0.001
        self.es_monitor = "val_loss"
        self.es_restore_best_weights = True

def main():
    args = Args()
    set_seed(args.seed)

    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 2. Model initialization
    model = FightDetection3DCNN(
        num_classes=args.num_classes,
        hidden_size=args.hidden_size,
        dropout_prob=args.dropout_prob,
    )
    model.to(device)

    # 3. Loss function
    criterion = nn.CrossEntropyLoss()

    # 4. Optimizer
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported.")

    # 5. Data preparation
    if not os.path.exists(args.data_preprocessed_dir):
        print(f"Error: Dataset directory not found at {args.data_preprocessed_dir}")
        return

    # Tạo SegmentationDataset với YOLO segmentation
    full_dataset = SegmentationDataset(
        video_dir=args.data_preprocessed_dir,
        classes_list=['fight', 'normal'],
        model_path="yolo11n-seg.pt",
        target_class="person",
        sequence_length=args.sequence_length,
        image_height=args.image_height,
        image_width=args.image_width,
    )

    print(f"Tổng số video tìm thấy: {len(full_dataset)}")
    
    if len(full_dataset) == 0:
        print("Error: No video files found in the dataset directory. Exiting.")
        return

    # Split the dataset into training and validation
    train_size = int((1 - args.val_split) * len(full_dataset)) 
    val_size = len(full_dataset) - train_size

    # Ensure val_size is at least 1 if total samples allow
    if train_size == 0 and len(full_dataset) > 0:
        train_size = 1
        val_size = len(full_dataset) - 1
    elif val_size == 0 and len(full_dataset) > 1:
        val_size = 1
        train_size = len(full_dataset) - 1

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Giảm num_workers vì segmentation cần nhiều RAM
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    print(f"Số video huấn luyện: {len(train_dataset)}")
    print(f"Số video validation: {len(val_dataset)}")

    # Tạo thư mục weights nếu chưa có
    model_dir = os.path.dirname(args.model_save_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 6. Callbacks
    callbacks = []

    reduce_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min" if args.lr_reduce_monitor == "val_loss" else "max",
        factor=args.lr_reduce_factor,
        patience=args.lr_reduce_patience,
        min_lr=args.lr_reduce_min_lr,
    )
    callbacks.append(reduce_lr_scheduler)

    # 7. Initialize and run Trainer
    trainer = Trainer3DCNN(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        callbacks=callbacks,
        model_save_path=args.model_save_path,
    )

    print("Bắt đầu huấn luyện với segmentation data...")
    model_3dcnn = trainer.train(num_epochs=args.epochs)

    print("\nLịch sử huấn luyện cuối cùng:")
    for key, values in model_3dcnn.items():
        print(f"{key}: {values}")

    # 8. Plotting the results
    plot_combined_metrics(model_3dcnn)

if __name__ == "__main__":
    # Kiểm tra data_organized có tồn tại không
    if not os.path.exists("data_organized"):
        print("ERROR: Chưa có thư mục data_organized!")
        print("Hãy chạy cell tổ chức data trước")
        exit()

    print("Bắt đầu training với segmentation data...")
    
    # Chạy training
    main()
