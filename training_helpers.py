import torch
class EarlyStopping():
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model, model_path):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # 保存模型
            torch.save(model.state_dict(), model_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True