import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class CustomDataset(Dataset):
    def __init__(self, folder):
        self.pkl_file_paths = [os.path.join(folder, file) for file in os.listdir(folder)]

    def __len__(self):
        return len(self.pkl_file_paths)

    def __getitem__(self, idx):
        file_path = self.pkl_file_paths[idx]
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        return data
    
def collate_fn(batch):
    collated_batch = {}
    for key in list(batch[0].keys()):
        if key.endswith('feature'):
            features = [torch.from_numpy(item[key]) for item in batch]
            features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
            collated_batch[key] = features_padded
        else:
            collated_batch[key] = [item[key] for item in batch]
    
    media_mask = np.where(np.all(collated_batch['visual_feature'].numpy() == 0, axis=2), True, False)
    media_mask = torch.from_numpy(media_mask)
    
    genres_mask = np.where(np.all(collated_batch['genres_feature'].numpy() == 0, axis=2), True, False)
    genres_mask = torch.from_numpy(genres_mask)
    collated_batch['media_mask'] = media_mask
    collated_batch['genres_mask'] = genres_mask
    
    return collated_batch


# pkl_files = ['tt0006864.pkl', 'tt0015724.pkl']
# dataset = CustomDataset(pkl_files)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# for batch in dataloader:
#     print(batch.keys())
    