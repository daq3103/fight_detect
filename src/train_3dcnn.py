import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import os  # Thêm import os để kiểm tra đường dẫn

from configs.configs import parse_arguments
from models.model_3dcnn_r2plus1d import FightDetection3DCNN
from data.datasets import  VideoDataset 

from utils.callbacks import EarlyStopping
from trainer.trainer import Trainer3DCNN
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
    model = FightDetection3DCNN(
        num_classes=args.num_classes,
        hidden_size=args.hidden_size,
        dropout_prob=args.dropout_prob,
    )
    model.to(device)

    # Optional: Plot model structure
    # Uncomment if you have graphviz installed and want to generate plot
    # model.plot_model_structure(input_shape=(1, args.sequence_length, args.img_channels, args.image_height, args.image_width))
    # print(f"Model structure plot saved to MobBiLSTM_model_structure_plot.png")

    # 3. Loss function
    criterion = nn.CrossEntropyLoss()

    # 4. Optimizer
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported.")

    if not os.path.exists(args.data_preprocessed_dir):
        print(f"Error: Dataset directory not found at {args.data_preprocessed_dir}")
        print(
            "Please ensure video files are located in the specified directory."
        )
        return  # Exit if data directory is not found

    full_dataset = VideoDataset(
        data_dir=args.data_preprocessed_dir,
        classes_list=['fight', 'no_fi'],
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
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    print(f"Số video huấn luyện: {len(train_dataset)}")
    print(f"Số video validation: {len(val_dataset)}")

    model_dir = os.path.dirname(args.model_save_path)
    # Nếu đường dẫn không rỗng và thư mục chưa tồn tại, hãy tạo nó
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 6. Callbacks
    callbacks = []

    # early_stopping_callback = EarlyStopping(
    #     patience=args.es_patience,
    #     verbose=args.es_verbose,
    #     delta=args.es_delta,
    #     path=args.model_save_path,
    #     monitor=args.es_monitor,
    #     restore_best_weights=args.es_restore_best_weights,
    # )
    # callbacks.append(early_stopping_callback)

    reduce_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min" if args.lr_reduce_monitor == "val_loss" else "max",
        factor=args.lr_reduce_factor,
        patience=args.lr_reduce_patience,
        min_lr=args.lr_reduce_min_lr,
        verbose=args.lr_reduce_verbose,
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
    main()
