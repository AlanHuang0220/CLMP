import pickle
from Dataset import CustomDataset, collate_fn
from torch.utils.data import DataLoader
from Model import CLMP
import torch
import torch.nn.functional as F
import clip


pkl_files = ['tt0006864.pkl', 'tt0015724.pkl']
dataset = CustomDataset(pkl_files)
dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# Constants
EMBEDDING_SIZE = 512
FF_HIDDEN_DIM = 1024
NUM_HEADS = 4
NUM_BLOCKS = 2


device = torch.device("cuda:1")

model = CLMP(EMBEDDING_SIZE, NUM_HEADS, FF_HIDDEN_DIM, NUM_BLOCKS, device=device, dropout=0.1)
model.to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

clip_model, preprocess = clip.load("ViT-B/32", device="cuda:1")
text_list = [
    "The movie's genre are Drama",
    "The movie's genre are Love Story",
    "The movie's genre are Action",
    "The movie's genre are Mystery",
    "The movie's genre are History",
    "The movie's genre are Sci-Fi",
    "The movie's genre are Animation",
    "The movie's genre are Crime",
    "The movie's genre are Youth",
    "The movie's genre are Horror",
    "The movie's genre are Thriller",
    "The movie's genre are Adventure",
    "The movie's genre are Gangster",
    "The movie's genre are War",
    "The movie's genre are Comedy",
    "The movie's genre are Documentary",
    "The movie's genre are Musical",
    "The movie's genre are Romance"
]
text = clip.tokenize(text_list).to(device)

with torch.no_grad():
    for batch in dataloader:
        visual_feature = batch['visual_feature'].to(device)  # Assuming these keys exist in your dataset
        audio_feature = batch['audio_feature'].to(device)
        # genres_feature = batch['genres_feature'].to(device)
        # print(genres_feature[0].shape)
        print(batch['genres_text'])
        
        mask = batch['media_mask'].to(device)
                
        visual_output, audio_output, va_fusion_output = model(visual_feature, audio_feature, mask)
        text_output = clip_model.encode_text(text).to(torch.float32)
        visual_output = F.normalize(visual_output, dim=1)
        text_output = F.normalize(text_output, dim=1)
        va_fusion_output = F.normalize(va_fusion_output, dim=1)
        # genres_feature = F.normalize(genres_feature, dim=2)
        cosine_similarities = torch.mm(va_fusion_output, text_output.to(device).t())

        # sorted_indices = torch.argsort(cosine_similarities, descending=True)
        # sorted_similarities = cosine_similarities[sorted_indices]
        print(cosine_similarities)