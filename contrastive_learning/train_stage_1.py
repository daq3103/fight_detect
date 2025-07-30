# train_stage_1_contrastive.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os

from models.model_3dcnn import FightDetector3DCNN
from data.dataset import ContrastiveVideoDataset, collate_fn
from utils.losses import NTXentLoss
from configs.default_config import STAGE1_CL_CONFIG, DEVICE, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, KINETICS_MEAN, KINETICS_STD

def main():
    print("--- Bắt đầu Giai đoạn 1: Contrastive Learning ---")
    config = STAGE1_CL_CONFIG
    os.makedirs(config['save_dir'], exist_ok=True)

    # 1. Augmentation mạnh cho Contrastive Learning
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(IMAGE_HEIGHT, IMAGE_WIDTH), scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.Normalize(mean=KINETICS_MEAN, std=KINETICS_STD),
    ])

    # 2. DataLoader
    # Gộp cả train và val để có nhiều dữ liệu hơn, vì ta không cần nhãn
    train_path = os.path.join(config['data_path'], 'train')
    val_path = os.path.join(config['data_path'], 'val')
    
    dataset = ContrastiveVideoDataset(
        data_dir=train_path, # Chỉ dùng train hoặc gộp cả train/val
        sequence_length=SEQUENCE_LENGTH,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        transform=transform
    )
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    print(f"Đã tải {len(dataset)} mẫu video cho Contrastive Learning.")

    # 3. Model, Loss, Optimizer
    model = FightDetector3DCNN().to(DEVICE)
    criterion = NTXentLoss(temperature=config['temperature'], device=DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(data_loader)*config['epochs'])

    # 4. Training Loop
    best_loss = float('inf')
    for epoch in range(config['epochs']):
        running_loss = 0.0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch in progress_bar:
            if not batch: continue
            view1, view2 = batch
            view1, view2 = view1.to(DEVICE), view2.to(DEVICE)
            
            optimizer.zero_grad()
            
            emb1, emb2 = model((view1, view2), mode='contrastive')
            loss = criterion(emb1, emb2)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best_model.pt'))
            print(f"🎉 Model mới tốt nhất được lưu với loss: {best_loss:.4f}")
            
    print("Hoàn tất Giai đoạn 1 - Contrastive Learning.")

if __name__ == "__main__":
    main()