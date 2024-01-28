import torch
import torch.nn as nn
import torch.optim as optim
from Model import CLMP ,CLMPWithHead
from Dataset import CustomDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from training_helpers import EarlyStopping
from Metrics import Top1Accuracy

def normalization(input, max, min):
    output = (input - min) / (max - min)
    return output

def test(task, model, test_loader, metric, device):
    model.eval()
    result = 0
    with torch.no_grad():
        for batch in test_loader:
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
            result += metric(output, label)
            
    return result / len(test_loader)

device = torch.device("cuda:1")
dataset_root = '/home/miislab-server/Alan/Alan_shared/LVU/feature'

pretrained_model = CLMP()

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
    testing_dataset = CustomDataset(f"{dataset_root}/{task_name}/test")
    
    testing_dataloader = DataLoader(dataset=testing_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    model = CLMPWithHead(pretrained_model, output_dim=task_info['output_dim'])
    model_weight = torch.load(f'finetune_weight/{task_name}_best_val.pth')
    model.load_state_dict(model_weight)
    model.to(device)
    
     # 定義 評估指標
    if task_info['type'] == 'classification':
        metric = Top1Accuracy()
    else:
        metric = nn.MSELoss()
    
    test_result = test(task_name, model, testing_dataloader, metric, device)

    print(f"task_name: {task_name}, result: {test_result}")
            