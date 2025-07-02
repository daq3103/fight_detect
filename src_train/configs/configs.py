# configs/configs.py
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fight Detection Model Training")

    # Model parameters
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes.')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size for LSTM layer.')
    parser.add_argument('--dropout_prob', type=float, default=0.5, help='Dropout probability.')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate for optimizer.')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'], help='Optimizer to use.')

    # Early Stopping Callback parameters
    parser.add_argument('--es_patience', type=int, default=10, help='Early stopping patience.')
    parser.add_argument('--es_monitor', type=str, default='val_accuracy', choices=['val_loss', 'val_accuracy'], help='Metric to monitor for early stopping.')
    parser.add_argument('--es_verbose', type=bool, default=True, help='Print message on early stopping improvement.')
    parser.add_argument('--es_delta', type=float, default=0.0, help='Minimum change to qualify as an improvement for early stopping.')
    parser.add_argument('--es_restore_best_weights', type=bool, default=True, help='Restore best model weights on early stopping.')

    # ReduceLROnPlateau Callback parameters
    parser.add_argument('--lr_reduce_monitor', type=str, default='val_loss', choices=['val_loss', 'val_accuracy'], help='Metric to monitor for LR reduction.')
    parser.add_argument('--lr_reduce_factor', type=float, default=0.6, help='Factor by which the learning rate will be reduced.')
    parser.add_argument('--lr_reduce_patience', type=int, default=5, help='Number of epochs with no improvement after which learning rate will be reduced.')
    parser.add_argument('--lr_reduce_min_lr', type=float, default=0.00005, help='Lower bound on the learning rate.')
    parser.add_argument('--lr_reduce_verbose', type=bool, default=True, help='Print message on LR reduction.')

    # Data parameters (Updated)/kaggle/input/real-life-violence-situations-dataset/Real Life Violence Dataset
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/real-life-violence-situations-dataset/', help='Root directory of the dataset.')
    parser.add_argument('--image_height', type=int, default=64, help='Height of video frames after resizing.')
    parser.add_argument('--image_width', type=int, default=64, help='Width of video frames after resizing.')
    parser.add_argument('--sequence_length', type=int, default=16, help='Number of frames to extract per video.')
    parser.add_argument('--classes_list', nargs='+', default=["NonViolence", "Violence"], help='List of class names.')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio.') # train_test_split in Keras was 0.1, now using val_split for random_split

    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--model_save_path', type=str, default='best_mobibilstm_model.pt', help='Path to save the best model.')

    args = parser.parse_args()
    return args