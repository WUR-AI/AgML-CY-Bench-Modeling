import copy
import logging
import numpy as np

log = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False, trace_func=log.info):
        """
        Args:
            patience (int): How many checks to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
            trace_func (function): trace print function.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.trace_func = trace_func
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        self.best_epoch: int | None = None

    def reset(self) -> None:
        """Clear state so the same instance can be reused across fits."""
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        self.best_epoch = None

    def __call__(self, val_loss, model, epoch: int):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint_in_memory(val_loss, model, epoch)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint_in_memory(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint_in_memory(self, val_loss, model, epoch: int):
        '''Saves model state to memory.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Caching best model...')
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.best_epoch = epoch