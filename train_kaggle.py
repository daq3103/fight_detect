# Kaggle version
import sys
import os

# Kaggle specific paths
sys.path.append('/kaggle/working/FightDetection/src')

# Import cần thiết
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import random

from src.configs.configs import parse_arguments
from src.models.model_3dcnn_r2plus1d import FightDetection3DCNN
from src.data.datasets import SegmentationDataset
from src.trainer.trainer import Trainer3DCNN
from src.utils.viz import plot_combined_metrics

class Args:
    def __init__(self):
        # Kaggle paths
        self.data_preprocessed_dir = "/kaggle/working/FightDetection/data_organized"
        self.model_save_path = "/kaggle/working/segmented_model.pth"
        
        # Model params - tăng để train thực sự
        self.num_classes = 2
        self.sequence_length = 30
        self.image_height = 224
        self.image_width = 224
        self.img_channels = 3
        self.hidden_size = 256  # Tăng lên
        self.dropout_prob = 0.3
        self.learning_rate = 0.0001  # Giảm xuống
        self.optimizer = "AdamW"
        self.batch_size = 4  # Kaggle GPU limit
        self.epochs = 50  # Train thật
        self.val_split = 0.2
        self.seed = 42
        
        # Scheduler
        self.lr_reduce_monitor = "val_loss"
        self.lr_reduce_factor = 0.5
        self.lr_reduce_patience = 5
        self.lr_reduce_min_lr = 1e-7

def main():
    args = Args()
    
    print("=== KAGGLE SEGMENTATION TRAINING ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Model
    model = FightDetection3DCNN(
        num_classes=args.num_classes,
        hidden_size=args.hidden_size,
        dropout_prob=args.dropout_prob,
    )
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss và optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Dataset
    print("\nLoading dataset...")
    full_dataset = SegmentationDataset(
        video_dir=args.data_preprocessed_dir,
        classes_list=['fight', 'normal'],
        model_path="yolo11n-seg.pt",
        target_class="person",
        sequence_length=args.sequence_length,
        image_height=args.image_height,
        image_width=args.image_width,
    )

    print(f"Total videos: {len(full_dataset)}")
    
    # Split dataset
    train_size = int((1 - args.val_split) * len(full_dataset)) 
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Training videos: {len(train_dataset)}")
    print(f"Validation videos: {len(val_dataset)}")

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_reduce_factor, 
        patience=args.lr_reduce_patience, min_lr=args.lr_reduce_min_lr
    )

    # Trainer
    trainer = Trainer3DCNN(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        callbacks=[scheduler],
        model_save_path=args.model_save_path,
    )

    print("\n🚀 Starting training...")
    history = trainer.train(num_epochs=args.epochs)

    print("\n✅ Training completed!")
    print("Training history:")
    for key, values in history.items():
        print(f"{key}: {values[-1]:.4f}")

    # Plot results
    try:
        plot_combined_metrics(history)
        print("📊 Metrics plotted!")
    except:
        print("⚠️ Could not plot metrics")

    return history

if __name__ == "__main__":
    history = main()