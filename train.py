from Dataset import CustomDataset, collate_fn
from torch.utils.data import DataLoader
from Model import CLMP
from Loss import DescriptionWithGenresContrastiveLoss
from  torch import optim
import os
import torch
from tqdm import tqdm
# tqdm.monitor_interval = 0


# Constants
BATCH_SIZE = 64
EMBEDDING_SIZE = 512
FF_HIDDEN_DIM = 1024
NUM_HEADS = 4
NUM_BLOCKS = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 100000

device = torch.device("cuda:1")

dataset_folder = '/home/miislab-server/Alan/Alan_shared/MovieNet/feature'
# pkl_files = [os.path.join(dataset_folder, file) for file in os.listdir(dataset_folder)]

dataset = CustomDataset(dataset_folder)
dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

model = CLMP(EMBEDDING_SIZE, NUM_HEADS, FF_HIDDEN_DIM, NUM_BLOCKS, dropout=0.1)
model.to(device)

dg_contrastive_loss = DescriptionWithGenresContrastiveLoss()
# dg_contrastive_loss = GenreContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

best_loss = float('Inf')
for epoch in range(NUM_EPOCHS):
    progress_bar = tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1}')
    model.train()
    running_loss = 0.0
    for batch in dataloader:
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
    average_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {average_loss:.4f}")
    print('Learning Rate', optimizer.param_groups[0]['lr'])
    if average_loss < best_loss:
        best_loss = average_loss
        torch.save(model.state_dict(), "bast_model.pth")

torch.save(model.state_dict(), "last_model.pth")