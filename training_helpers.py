import torch
class EarlyStopping():
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        self.early_stop = False
        self.counter = 0

    def __call__(self, train_loss, val_loss, model, model_path):
        # 保存 val loss 最低的模型
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), model_path + '_best_val.pth')
        else:
            self.counter += 1

        # 保存train loss 最低的模型
        if train_loss < self.best_train_loss - self.min_delta:
            self.best_train_loss = train_loss
            torch.save(model.state_dict(), model_path + '_best_train.pth')

        
        if self.counter >= self.patience:
            self.early_stop = True
            torch.save(model.state_dict(), model_path + '_final.pth')