# configs/default_config.py
import torch

# --- Cấu hình chung ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES_LIST = ["NonFight", "Fight"]
NUM_CLASSES = len(CLASSES_LIST)
MODEL_SAVE_DIR = "./checkpoints"

# --- Cấu hình tiền xử lý & DataLoader ---
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
SEQUENCE_LENGTH = 16

# --- Giai đoạn 1: Contrastive Learning ---
STAGE1_CL_CONFIG = {
    "data_path": "/kaggle/input/rwf2000/RWF-2000", # Có thể dùng cả train/val vì không cần nhãn
    "epochs": 50, # CL cần nhiều epoch hơn
    "batch_size": 32, # Batch size lớn hơn tốt cho CL
    "learning_rate": 3e-4,
    "temperature": 0.07, # Tham số cho NT-Xent Loss
    "save_dir": f"{MODEL_SAVE_DIR}/stage_1_contrastive",
}

# --- Giai đoạn 2: Supervised Fine-tuning ---
STAGE2_SUPERVISED_CONFIG = {
    "data_path": "/kaggle/input/rwf2000/RWF-2000",
    "stage1_best_model_path": f"{STAGE1_CL_CONFIG['save_dir']}/best_model.pt",
    "epochs": 15,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "unfreeze_layers": 0, # Chỉ train classifier, không unfreeze backbone
    "save_dir": f"{MODEL_SAVE_DIR}/stage_2_supervised",
}