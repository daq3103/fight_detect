import torch
import numpy as np
import copy


class EarlyStopping:
    """
    Early stops the training if validation accuracy doesn't improve after a given patience.
    """
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, monitor='val_accuracy', restore_best_weights=True):
        """
        Args:
            patience (int): How long to wait after last time validation accuracy improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation accuracy improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            path (str): Path for the checkpoint to be saved to.
                        Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                                   Default: print
            monitor (str): Metric to monitor. 'val_accuracy' or 'val_loss'.
            restore_best_weights (bool): If True, restores model weights from the epoch
                                         with the best monitored metric.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_max = -np.Inf if monitor == 'val_accuracy' else np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.monitor = monitor
        self.restore_best_weights = restore_best_weights
        self.best_model_state = None

    def __call__(self, val_metric, model):
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif (self.monitor == 'val_accuracy' and score < self.best_score + self.delta) or \
             (self.monitor == 'val_loss' and score > self.best_score - self.delta):
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        '''Saves model when validation metric improves.'''
        if self.verbose:
            if self.monitor == 'val_accuracy':
                self.trace_func(f'Validation accuracy improved ({self.val_metric_max:.6f} --> {val_metric:.6f}). Saving model ...')
            else: # val_loss
                self.trace_func(f'Validation loss decreased ({self.val_metric_max:.6f} --> {val_metric:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.best_model_state = copy.deepcopy(model.state_dict()) # Store the best state
        self.val_metric_max = val_metric