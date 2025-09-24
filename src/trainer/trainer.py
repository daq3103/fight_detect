import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm # Để hiển thị thanh tiến trình
import numpy as np
import time
import os
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from utils.callbacks import EarlyStopping

class Trainer_CNN_LSTM:
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
    

class Trainer3DCNN:
    def __init__(self, model, criterion, optimizer, device,
                 train_dataloader, val_dataloader,
                 callbacks=None, model_save_path='./weights/best_model.pt',
                 use_amp=True, grad_accum_steps=2):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.callbacks = callbacks if callbacks is not None else []
        self.model_dir = model_save_path
        self.use_amp = use_amp
        self.grad_accum_steps = grad_accum_steps 
        self.scaler = GradScaler(enabled=use_amp)
        
        os.makedirs(model_save_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(model_save_path, 'logs'))
        
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'epoch_time': [], 'learning_rate': []
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0

        # Setup callbacks
        self.early_stopping = None
        self.lr_scheduler = None
        for callback in self.callbacks:
            if isinstance(callback, EarlyStopping):
                self.early_stopping = callback
            elif isinstance(callback, torch.optim.lr_scheduler._LRScheduler):
                self.lr_scheduler = callback

    def _train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_time = time.time()
        
        optimizer = self.optimizer
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (inputs, labels) in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch+1} Train")):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Mixed precision forward
            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss = loss / self.grad_accum_steps  # Scale loss for gradient accumulation
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            # Metrics calculation
            running_loss += loss.item() * inputs.size(0) * self.grad_accum_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Free GPU memory immediately
            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc, time.time() - batch_time

    def _validate(self, num_epochs):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_dataloader, desc=f"Epoch {num_epochs+1} Val"):
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                del inputs, labels, outputs
                torch.cuda.empty_cache()
        
        val_loss /= total
        val_acc = correct / total
        return val_loss, val_acc

    def train(self, num_epochs):
        print(f"Training started on {self.device}")
        print(f"Using AMP: {self.use_amp} | Grad Accumulation: {self.grad_accum_steps} steps")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train phase
            train_loss, train_acc, epoch_time = self._train_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc = self._validate(epoch)
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_time'].append(epoch_time)
            self.history['learning_rate'].append(current_lr)
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning Rate', current_lr, epoch)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(self.model_dir, 'best_model.pth'))
            
            # Save latest checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(self.model_dir, 'last_checkpoint.pth'))
            
            # Early stopping
            stop_training = False
            if self.early_stopping:
                self.early_stopping(val_loss, self.model)
                if self.early_stopping.early_stop:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    stop_training = True
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Best Acc: {self.best_val_acc:.4f} @ epoch {self.best_epoch+1}")
            print(f"LR: {current_lr:.6f}")
            
            if stop_training:
                break
        
        self.writer.close()
        print(f"\nTraining completed. Best val accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}")
        return self.history

class TrainerSTGCN:
    def __init__(self, model, criterion, optimizer, device,
                 train_dataloader, val_dataloader,
                 callbacks=None, model_save_path='./weights/best_model.pt',
                 use_amp=True, grad_accum_steps=2):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.callbacks = callbacks if callbacks is not None else []
        self.model_dir = model_save_path
        self.use_amp = use_amp
        self.grad_accum_steps = grad_accum_steps
        self.scaler = GradScaler(enabled=use_amp)
        
        os.makedirs(model_save_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(model_save_path, 'logs'))
        
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'epoch_time': [], 'learning_rate': []
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0

        # Setup callbacks
        self.early_stopping = None
        self.lr_scheduler = None
        for callback in self.callbacks:
            if isinstance(callback, EarlyStopping):
                self.early_stopping = callback
            elif isinstance(callback, torch.optim.lr_scheduler._LRScheduler):
                self.lr_scheduler = callback

    def _train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_time = time.time()
        
        optimizer = self.optimizer
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (inputs, labels) in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch+1} Train")):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Mixed precision forward
            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss = loss / self.grad_accum_steps  # Scale loss for gradient accumulation
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            # Metrics calculation
            running_loss += loss.item() * inputs.size(0) * self.grad_accum_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Free GPU memory immediately
            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc, time.time() - batch_time

    def _validate(self, num_epochs):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_dataloader, desc=f"Epoch {num_epochs+1} Val"):
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                del inputs, labels, outputs
                torch.cuda.empty_cache()
        
        val_loss /= total
        val_acc = correct / total
        return val_loss, val_acc

    def train(self, num_epochs):
        print(f"Training started on {self.device}")
        print(f"Using AMP: {self.use_amp} | Grad Accumulation: {self.grad_accum_steps} steps")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train phase
            train_loss, train_acc, epoch_time = self._train_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc = self._validate(epoch)
            ###
            if self.lr_scheduler:
                # Dùng val_loss cho ReduceLROnPlateau
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                # Các scheduler khác không cần tham số
                else:
                    self.lr_scheduler.step()
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['learning_rate'].append(current_lr)
            self.writer.add_scalar('Learning Rate', current_lr, epoch)
                
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_time'].append(epoch_time)
            self.history['learning_rate'].append(current_lr)
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning Rate', current_lr, epoch)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(self.model_dir, 'best_model.pth'))
            
            # Save latest checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(self.model_dir, 'last_checkpoint.pth'))
            
            # Early stopping
            stop_training = False
            if self.early_stopping:
                self.early_stopping(val_loss, self.model)
                if self.early_stopping.early_stop:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    stop_training = True
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Best Acc: {self.best_val_acc:.4f} @ epoch {self.best_epoch+1}")
            print(f"LR: {current_lr:.6f}")
            
            if stop_training:
                break
        
        self.writer.close()
        print(f"\nTraining completed. Best val accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}")
        return self.history
    


