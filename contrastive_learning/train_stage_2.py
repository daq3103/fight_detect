# train_stage_2_supervised.py
# (File này gần như giống với file train_stage_2.py ở phiên bản trước,
# chỉ thay đổi tên config và logic prepare model cho rõ ràng)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Thay dataset cũ bằng SupervisedVideoDataset
from data.dataset import SupervisedVideoDataset, collate_fn 
from models.model_3dcnn import FightDetector3DCNN
from utils.trainer import SupervisedTrainer # Trainer này vẫn dùng tốt
# Thay đổi config import
from configs.default_config import STAGE2_SUPERVISED_CONFIG as config, DEVICE, CLASSES_LIST, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH

def main():
    print("--- Bắt đầu Giai đoạn 2: Supervised Fine-tuning ---")
    
    # 1. DataLoader
    train_dataset = SupervisedVideoDataset(
        data_dir=f"{config['data_path']}/train",
        classes_list=CLASSES_LIST,
        sequence_length=SEQUENCE_LENGTH,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH
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
    optimizer = optim.AdamW(model.classifier.parameters(), lr=config['learning_rate'])
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