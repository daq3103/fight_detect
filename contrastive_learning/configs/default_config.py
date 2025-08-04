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

# --- Đặc điểm của phân phối dữ liệu Pretrain --- 

KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]

# --- Giai đoạn 1: Contrastive Learning ---
STAGE1_CL_CONFIG = {
    "data_path": "/kaggle/input/rwf2000/RWF-2000",
    "epochs": 70, # CL cần nhiều epoch hơn
    "batch_size": 16, # Batch size lớn hơn tốt cho CL
    "learning_rate": 5e-5,
    "temperature": 0.07, # Tham số cho NT-Xent Loss
    "save_dir": f"{MODEL_SAVE_DIR}/stage_1_contrastive",
    "virtual_batch_size": 128,
    "warmup_epochs": 5
}

# --- Giai đoạn 2: Supervised Fine-tuning ---
STAGE2_SUPERVISED_CONFIG = {
    "data_path": "/kaggle/input/violencedataset/mix_data",
    "stage1_best_model_path": f"/kaggle/nput/best_cl_model/pytorch/default/1/best_CL_model.pt",
    "epochs": 30,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "unfreeze_layers": 0, # Chỉ train classifier, không unfreeze backbone
    "save_dir": f"{MODEL_SAVE_DIR}/stage_2_supervised",
}