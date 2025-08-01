# train_stage_1_contrastive.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2 as T
from tqdm import tqdm
import os
import random
from models.model_3dcnn import FightDetector3DCNN
from data.dataset import SemanticContrastiveDataset, semantic_collate_fn
from utils.losses import TripletLoss
from configs.default_config import (
    STAGE1_CL_CONFIG,
    DEVICE,
    SEQUENCE_LENGTH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)


def temporal_crop_fn(vid, ratio=0.8):
    T0 = vid.size(0)
    L = max(1, int(T0 * ratio))
    start = random.randint(0, T0 - L)
    return vid[start : start + L]


def main():
    print("--- Bắt đầu Giai đoạn 1: Contrastive Learning ---")
    config = STAGE1_CL_CONFIG
    os.makedirs(config["save_dir"], exist_ok=True)

    # 1. Augmentation mạnh cho Contrastive Learning
    transform = T.Compose(
        [
            T.Lambda(lambda vid: temporal_crop_fn(vid, 0.8)),
            T.Lambda(lambda vid: vid.flip(0) if random.random() < 0.3 else vid),
            T.Lambda(
                lambda vid: torch.stack(
                    [T.GaussianBlur(kernel_size=5)(f) for f in vid], dim=0
                )
            ),
            T.Lambda(
                lambda vid: torch.stack(
                    [T.RandomResizedCrop((224, 224), scale=(0.8, 1.0))(f) for f in vid],
                    dim=0,
                )
            ),
            T.Lambda(
                lambda vid: torch.stack(
                    [T.ColorJitter(0.2, 0.2, 0.2, 0.1)(f) for f in vid], dim=0
                )
            ),
        ]
    )

    # 2. DataLoader
    # Gộp cả train và val để có nhiều dữ liệu hơn, vì ta không cần nhãn
    train_path = os.path.join(config["data_path"], "train")
    val_path = os.path.join(config["data_path"], "val")

    dataset = SemanticContrastiveDataset(
        data_dir=train_path,
        sequence_length=SEQUENCE_LENGTH,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        transform=transform,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=semantic_collate_fn,
        num_workers=4,
    )
    print(f"Đã tải {len(dataset)} mẫu video cho Contrastive Learning.")

    # 3. Model, Loss, Optimizer
    model = FightDetector3DCNN().to(DEVICE)
    if torch.cuda.device_count() > 1:
        print(f"Sử dụng {torch.cuda.device_count()} GPU!")
        model = nn.DataParallel(model, device_ids=[0, 1])
    criterion = TripletLoss(temperature=config["temperature"], device=DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=1e-6
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(data_loader) * config["epochs"]
    )

    # 4. Training Loop
    best_loss = float("inf")
    for epoch in range(config["epochs"]):
        running_loss = 0.0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")

        for batch in progress_bar:
            if not batch:
                continue
            anchors = batch["anchors"].to(DEVICE)
            positives = batch["positives"].to(DEVICE)
            negatives = batch["negatives"].to(DEVICE)

            optimizer.zero_grad()

            # Forward 3 loại video
            emb_anchors = model(anchors, mode="contrastive")
            emb_positives = model(positives, mode="contrastive")
            emb_negatives = model(negatives, mode="contrastive")

            # Tính triplet loss
            loss = criterion(emb_anchors, emb_positives, emb_negatives)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                model.state_dict(), os.path.join(config["save_dir"], "best_model.pt")
            )
            print(f"🎉 Model mới tốt nhất được lưu với loss: {best_loss:.4f}")

    print("Hoàn tất Giai đoạn 1 - Contrastive Learning.")


if __name__ == "__main__":
    main()
