import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm # Để hiển thị thanh tiến trình
import numpy as np

from utils.callbacks import EarlyStopping

class Trainer:
    def __init__(self, model, criterion, optimizer, device,
                 train_dataloader, val_dataloader,
                 callbacks=None, model_save_path='./weights/best_model.pt'):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.callbacks = callbacks if callbacks is not None else []
        self.model_save_path = model_save_path

        self.history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
        self.best_val_metric = -np.Inf 

        self.early_stopping = None
        self.reduce_lr_scheduler = None
        for callback in self.callbacks:
            if isinstance(callback, EarlyStopping):
                self.early_stopping = callback
            elif isinstance(callback, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.reduce_lr_scheduler = callback

        if self.early_stopping and self.early_stopping.monitor == 'val_loss':
            self.best_val_metric = np.Inf # Reset for val_loss monitoring

    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Sử dụng tqdm để hiển thị thanh tiến trình
        for inputs, labels in tqdm(self.train_dataloader, desc="Training"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        return epoch_loss, epoch_accuracy

    def _validate_epoch(self):
        self.model.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_dataloader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total_samples += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()

        epoch_val_loss = val_running_loss / val_total_samples
        epoch_val_accuracy = val_correct_predictions / val_total_samples
        return epoch_val_loss, epoch_val_accuracy

    def train(self, num_epochs):
        print(f"Bắt đầu huấn luyện trên {self.device}...")
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

            train_loss, train_accuracy = self._train_epoch()
            val_loss, val_accuracy = self._validate_epoch()

            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)

            if self.reduce_lr_scheduler:
                monitor_val = val_loss if self.reduce_lr_scheduler.mode == 'min' else val_accuracy
                self.reduce_lr_scheduler.step(monitor_val)

            if self.early_stopping:
                monitor_val_es = val_accuracy if self.early_stopping.monitor == 'val_accuracy' else val_loss
                self.early_stopping(monitor_val_es, self.model)
                if self.early_stopping.early_stop:
                    print(f"Epoch {epoch+1}/{num_epochs} - Early stopping triggered.")
                    if self.early_stopping.restore_best_weights and self.early_stopping.best_model_state:
                        self.model.load_state_dict(self.early_stopping.best_model_state)
                        print(f"Restored model weights from best epoch.")
                    break

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f} - "
                  f"Current LR: {current_lr:.6f}")

        print("Quá trình huấn luyện hoàn tất.")
        return self.history