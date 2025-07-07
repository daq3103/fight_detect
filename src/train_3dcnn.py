# main.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import os  # Thêm import os để kiểm tra đường dẫn

# Import from your project structure
from configs.configs import parse_arguments
from models.model_3dcnn_r2plus1d import FightDetection3DCNN
from data.datasets import VideoDataset  # Import VideoDataset

# from data.data_utils import frames_extraction # Not directly used here, but good to know it's in data_utils
from utils.callbacks import EarlyStopping
from trainer.trainer import Trainer, Trainer3DCNN
from utils.viz import plot_metric, plot_combined_metrics  # Import plotting functions


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
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported.")

    # 5. Data preparation (Using the new VideoDataset)
    if not os.path.exists(args.data_preprocessed_dir):
        print(f"Error: Dataset directory not found at {args.data_preprocessed_dir}")
        print(
            "Please ensure 'Real Life Violence Dataset' is extracted and located correctly."
        )
        return  # Exit if data directory is not found

    full_dataset = VideoDataset(
        data_dir=args.data_preprocessed_dir,
        classes_list=args.classes_list,
        image_height=args.image_height,
        image_width=args.image_width,
        sequence_length=args.sequence_length,
        # No transform specified for now, as normalization is done in frames_extraction
    )

    # Split the dataset into training and validation
    train_size = int((1 - args.val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Ensure val_size is at least 1 if total samples allow
    if train_size == 0 and len(full_dataset) > 0:
        train_size = 1
        val_size = len(full_dataset) - 1
    elif train_size == 0 and len(full_dataset) == 0:
        print("Error: No samples found in the dataset. Exiting.")
        return

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )  # num_workers for faster loading
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    print(f"Tổng số mẫu trong dataset: {len(full_dataset)}")
    print(f"Số mẫu huấn luyện: {len(train_dataset)}")
    print(f"Số mẫu validation: {len(val_dataset)}")

    # 6. Callbacks
    callbacks = []

    early_stopping_callback = EarlyStopping(
        patience=args.es_patience,
        verbose=args.es_verbose,
        delta=args.es_delta,
        path=args.model_save_path,
        monitor=args.es_monitor,
        restore_best_weights=args.es_restore_best_weights,
    )
    callbacks.append(early_stopping_callback)

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

    MobBiLSTM_model_history = trainer.train(num_epochs=args.epochs)

    print("\nLịch sử huấn luyện cuối cùng:")
    for key, values in MobBiLSTM_model_history.items():
        print(f"{key}: {values}")

    # 8. Plotting the results
    plot_combined_metrics(MobBiLSTM_model_history)


if __name__ == "__main__":
    main()
