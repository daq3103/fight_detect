# utils/trainer.py
import torch
import time
import os
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

class SupervisedTrainer:
    def __init__(self, model, criterion, optimizer, device,
                 train_dataloader, val_dataloader, lr_scheduler,
                 config):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.lr_scheduler = lr_scheduler
        self.config = config
        
        self.save_dir = config["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, 'logs'))
        
        self.scaler = GradScaler('cuda', enabled=(device == 'cuda'))
        self.best_val_acc = 0.0

    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        for inputs, labels in progress_bar:
            if inputs.shape[0] == 0:  # Kiểm tra batch size
                continue
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with autocast('cuda', enabled=(self.device == 'cuda')):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix(loss=loss.item(), acc=100.*correct/total)
            
        return running_loss / total, correct / total

    def _validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_dataloader, desc="Validating")
            for inputs, labels in progress_bar:
                if not inputs.numel(): continue

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                progress_bar.set_postfix(loss=loss.item(), acc=100.*correct/total)

        return val_loss / total, correct / total

    def train(self):
        print(f"Bắt đầu huấn luyện. Checkpoints sẽ được lưu tại: {self.save_dir}")
        for epoch in range(self.config['epochs']):
            print(f"\n--- Epoch {epoch+1}/{self.config['epochs']} ---")
            
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._validate_epoch()
            
            if self.lr_scheduler:
                self.lr_scheduler.step(val_loss)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning Rate', current_lr, epoch)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pt'))
                print(f"🎉 New best model saved with Val Acc: {val_acc:.4f}")
        
        self.writer.close()
        print("Huấn luyện hoàn tất.")