import torch
import torch.nn as nn
import torch.optim as optim
from Model import CLMP ,CLMPWithHead
from Dataset import CustomDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from training_helpers import EarlyStopping
from tensorboardX import SummaryWriter

def normalization(input, max, min):
    output = (input - min) / (max - min)
    return output

def train(task, model, dataloader, criterion, optimizer, progress_bar, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        visual_feature = batch['visual_feature'].to(device)
        audio_feature = batch['audio_feature'].to(device)
        mask = batch['media_mask'].to(device)
        
        if task == 'like_ratio':
            like_count = normalization(batch['like_count'], 853196, 18595)
            dislike_count = normalization(batch['dislike_count'], 18595, 359)
            label = torch.cat((like_count, dislike_count), dim=1).to(device).float()
        elif task == 'view_count':
            view_count = normalization(batch['view_count'], 103520351, 531)
            label = view_count.to(device).float()
        else:
            label = batch['class_id'].to(device).squeeze(1)
        # print(label)
        
        output = model(visual_feature, audio_feature, mask)
        # print(output)
        loss = criterion(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f'Gradient for {name}:')
        #         print(param.grad)
        
        progress_bar.update(1)
    return total_loss / len(dataloader)

def validate(task, model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            visual_feature = batch['visual_feature'].to(device)
            audio_feature = batch['audio_feature'].to(device)
            mask = batch['media_mask'].to(device)
            
            if task == 'like_ratio':
                like_count = normalization(batch['like_count'], 853196, 18595)
                dislike_count = normalization(batch['dislike_count'], 18595, 359)
                label = torch.cat((like_count, dislike_count), dim=1).to(device).float()
            elif task == 'view_count':
                view_count = normalization(batch['view_count'], 103520351, 531)
                label = view_count.to(device).float()
            else:
                label = batch['class_id'].to(device).squeeze(1)
            
            output = model(visual_feature, audio_feature, mask)
            loss = criterion(output, label)
            
            total_loss += loss.item()
    return total_loss / len(val_loader)
        

writer = SummaryWriter(log_dir='finetune_weight/all_tasks')
device = torch.device("cuda:1")
dataset_root = '/home/miislab-server/Alan/Alan_shared/LVU/feature'
total_epochs = 10000
# init pretrain model
pretrained_model = CLMP()
pretrained_model.load_state_dict(torch.load('pretrain_weight/2024-01-24_19-40-47/check_point.pth'))  # load pretrain weight

# 故定預訓練模型權重
for param in pretrained_model.parameters():
    param.requires_grad = False
    
tasks = {
    'relationship': {'output_dim': 4, 'type': 'classification'},
    'way_speaking': {'output_dim': 5, 'type': 'classification'},
    'scene': {'output_dim': 6, 'type': 'classification'},
    
    'director': {'output_dim': 10, 'type': 'classification'},
    'genre': {'output_dim': 4, 'type': 'classification'},
    'writer': {'output_dim': 10, 'type': 'classification'},
    'year': {'output_dim': 9, 'type': 'classification'},
    
    'like_ratio': {'output_dim': 2, 'type': 'regression'},
    'view_count': {'output_dim': 1, 'type': 'regression'},
}

for task_name, task_info in tasks.items():
    save_dir = f'finetune_weight/{task_name}'
    training_dataset = CustomDataset(f"{dataset_root}/{task_name}/train")
    val_dataset = CustomDataset(f"{dataset_root}/{task_name}/val")
    testing_dataset = CustomDataset(f"{dataset_root}/{task_name}/test")
    
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    testing_dataloader = DataLoader(dataset=testing_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    model = CLMPWithHead(pretrained_model, output_dim=task_info['output_dim'])
    model.to(device)
    
     # 定義 loss
    if task_info['type'] == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    early_stopper = EarlyStopping(patience=50)
    
    for epoch in range(total_epochs):
        with tqdm(total=len(training_dataloader), desc=f"Task: {task_name}, Epoch: {epoch + 1}/{total_epochs}") as pbar:
            train_loss = train(task_name, model, training_dataloader, criterion, optimizer, pbar, device)
            val_loss = validate(task_name, model, val_dataloader, criterion, device)
            writer.add_scalar(f'Loss/{task_name}/train', train_loss, epoch)
            writer.add_scalar(f'Loss/{task_name}/val', val_loss, epoch)
            print(f"train_loss: {train_loss}, val_loss: {val_loss}")
            
        early_stopper(train_loss, val_loss, model, save_dir)

        if early_stopper.early_stop:
            print(f"Early stopping triggered epoch:{epoch+1}")
            writer.close()
            break    

writer.close()
            