# train_stage_2_supervised.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import SupervisedVideoDataset, collate_fn 
from models.model_3dcnn import FightDetector3DCNN
from utils.trainer import SupervisedTrainer 
from torchvision.transforms import v2 as T
import random
from configs.default_config import STAGE2_SUPERVISED_CONFIG as config, DEVICE, CLASSES_LIST, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH

def temporal_crop_fn(vid, ratio=0.8):
    T0 = vid.size(0)
    L = max(1, int(T0 * ratio))
    start = random.randint(0, T0 - L)
    return vid[start : start + L]


def main():
    print("--- Bắt đầu Giai đoạn 2: Supervised Fine-tuning ---")

    # 1. Augmentation mạnh cho Contrastive Learning
    transform = T.Compose(
        [
            # T.RandomHorizontalFlip(p=0.3),
            # T.ColorJitter(brightness=0.1, contrast=0.1),
            # T.Lambda(lambda vid: temporal_crop_fn(vid, 0.9)),
            
        ]
    )
    
        
    # 1. DataLoader
    train_dataset = SupervisedVideoDataset(
        data_dir=f"{config['data_path']}/train",
        classes_list=CLASSES_LIST,
        sequence_length=SEQUENCE_LENGTH,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        # transform=transform
    )
    
    val_dataset = SupervisedVideoDataset(
        data_dir=f"{config['data_path']}/val",
        classes_list=CLASSES_LIST,
        sequence_length=SEQUENCE_LENGTH,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    print(f"Đã tải {len(train_dataset)} mẫu train và {len(val_dataset)} mẫu val.")

    # 2. Model
    model = FightDetector3DCNN(num_classes=len(CLASSES_LIST))
    print(f"Tải trọng số từ Giai đoạn 1: {config['stage1_best_model_path']}")
    model.load_state_dict(torch.load(config['stage1_best_model_path'], map_location=DEVICE), strict=False)
    
    # Chuẩn bị model cho Giai đoạn 2: Đóng băng backbone
    model.prepare_for_finetuning_classifier()

    # 3. Optimizer, Loss, Scheduler
    # Chỉ train các tham số của classifier_head
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(params, lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3, verbose=True)

    # 4. Trainer
    trainer = SupervisedTrainer( # SupervisedTrainer vẫn dùng lại được
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        lr_scheduler=scheduler,
        config=config,
    )
    
    trainer.train()

if __name__ == "__main__":
    main()