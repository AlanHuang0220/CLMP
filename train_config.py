import os
import yaml
import Loss
import Model
import shutil
import Dataset
import torch.utils.data 
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter
from training_helpers import EarlyStopping
from Dataset import collate_fn
from argument_parser import parse_arguments
from config_utils import extract_config_components, initialize_component

device = torch.device('cuda:1')

args = parse_arguments()
config_file_path = args.config_file


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 創建以日期和時間為名稱的子目錄
log_dir = os.path.join('logs', timestamp)
config_dir = os.path.join('training_weight', timestamp)
os.makedirs(config_dir, exist_ok=True)
shutil.copyfile(config_file_path, os.path.join(config_dir,'cofig.yaml'))

# 創建新的 SummaryWriter
writer = SummaryWriter(log_dir)

with open(config_file_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

component_keys = ['training_dataset', 'val_dataset', 'data_loader', 'model', 'optimizer', 'lr_scheduler', 'loss', 'trainer']

 # Extract the components from the configuration
config_components = extract_config_components(config, component_keys)

training_dataset = initialize_component(config_components['training_dataset'], Dataset, config_components['training_dataset']['args'])
val_dataset = initialize_component(config_components['val_dataset'], Dataset, config_components['val_dataset']['args'])

dataloader_args = config_components['data_loader']['args']
dataloader_args['dataset'] = training_dataset
dataloader_args['collate_fn'] = collate_fn
training_dataloader = initialize_component(config_components['data_loader'], torch.utils.data, dataloader_args)

dataloader_args = config_components['data_loader']['args']
dataloader_args['dataset'] = val_dataset
dataloader_args['collate_fn'] = collate_fn
val_dataloader = initialize_component(config_components['data_loader'], torch.utils.data, dataloader_args)

model_args = config_components['model']['args']
model = initialize_component(config_components['model'], Model, model_args)
if config_components['trainer']['args']['use_check_point']:
    model.load_state_dict(torch.load(config_components['trainer']['args']['check_point']))
model.to(device)

optimizer_args = config_components['optimizer']['args']
optimizer_args['params'] = model.parameters()  
optimizer = initialize_component(config_components['optimizer'], optim, optimizer_args)

lr_scheduler_args = config_components['lr_scheduler']['args']
lr_scheduler_args['optimizer'] = optimizer
lr_scheduler = initialize_component(config_components['lr_scheduler'], optim.lr_scheduler, lr_scheduler_args)

dg_contrastive_loss = initialize_component(config_components['loss'], Loss, config_components['loss']['args'])

early_stopper = EarlyStopping(patience=1000)

for epoch in range(config_components['trainer']['args']['epochs']):
    progress_bar = tqdm(total=len(training_dataloader), desc=f'Epoch {epoch + 1}')
    model.train()
    running_loss = 0.0
    
    for batch in training_dataloader:
        visual_feature = batch['visual_feature'].to(device)  # Assuming these keys exist in your dataset
        audio_feature = batch['audio_feature'].to(device)
        genres_feature = batch['genres_feature'].to(device).float()
        description_feature = batch['description_feature'].to(device).squeeze(1).float()
        genres_label = batch['label']
        mask = batch['media_mask'].to(device)
        
        visual_output, audio_output, va_fusion_output = model(visual_feature, audio_feature, mask)
        
        # loss = dg_contrastive_loss(visual_output, audio_output, va_fusion_output, description_feature, genres_feature)
        loss = dg_contrastive_loss(visual_output, audio_output, va_fusion_output, description_feature, genres_feature)
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        running_loss += loss.item()
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f'Gradient for {name}:')
        #         print(param.grad)
        
        progress_bar.update(1)
        
    # Print average loss for this epoch
    progress_bar.close()
    average_loss = running_loss / len(training_dataloader)
    print(f"Epoch [{epoch+1}/{config_components['trainer']['args']['epochs']}] - Loss: {average_loss:.4f}")
    
    writer.add_scalar('Loss/Training Loss', average_loss, epoch+1)
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch+1)
    
    # Validation phase
    model.eval()
    val_running_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            visual_feature = batch['visual_feature'].to(device)  # Assuming these keys exist in your dataset
            audio_feature = batch['audio_feature'].to(device)
            genres_feature = batch['genres_feature'].to(device).float()
            description_feature = batch['description_feature'].to(device).squeeze(1).float()
            genres_label = batch['label']
            mask = batch['media_mask'].to(device)
            
            visual_output, audio_output, va_fusion_output = model(visual_feature, audio_feature, mask)
            
            val_loss = dg_contrastive_loss(visual_output, audio_output, va_fusion_output, description_feature, genres_feature)
            val_running_loss += val_loss.item()
        
        val_average_loss = val_running_loss / len(val_dataloader)    
        print(f"Validation - Loss: {val_average_loss:.4f}")   
        writer.add_scalar('Loss/Validation Loss', val_average_loss, epoch+1)

    early_stopper(average_loss, val_average_loss, model, os.path.join(config_dir, 'model'))

    if early_stopper.early_stop:
        print(f"Early stopping triggered epoch:{epoch+1}")
        break
            