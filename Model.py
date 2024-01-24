import torch
import torch.nn as nn
import torch.nn.functional as F

class AddPositionalEmbedding(nn.Module):
    def __init__(self, max_len, embedding_dim):
        super(AddPositionalEmbedding, self).__init__()
        # Create positional embeddings matrix
        self.positional_embeddings = nn.Parameter(torch.randn(1, max_len, embedding_dim))

    def forward(self, input):
        """
        input: Tensor of size [batch_size, seq_len, enbedding_dim]
        """
        seq_len = input.size(1)
        # Get the positional embeddings for the required sequence length
        
        return input + self.positional_embeddings[:, :seq_len, :]

class FeedForwardNN(nn.Module):
    def __init__(self, embedding_dim, ff_hidden_dim):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.fc2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = FeedForwardNN(embedding_dim, ff_hidden_dim)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_output, _ = self.attention(x, x, x, mask)
        x = x + self.dropout(attention_output)
        x = self.norm1(x)

        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x

class CrossAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = FeedForwardNN(embedding_dim, ff_hidden_dim)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        attention_output, _ = self.attention(query, key, value, mask)
        query = query + self.dropout(attention_output)
        query = self.norm1(query)

        ff_output = self.feed_forward(query)
        query = query + self.dropout(ff_output)
        query = self.norm2(query)

        return query

class ExternalFusionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(ExternalFusionBlock, self).__init__()
        self.audio_guided_attention = CrossAttention(embedding_dim, num_heads, ff_hidden_dim, dropout)
        self.visual_guided_attention = CrossAttention(embedding_dim, num_heads, ff_hidden_dim, dropout)
        
    def forward(self, visual, audio, mask):
        visual_output = self.audio_guided_attention(visual, audio, audio, mask)
        audio_output = self.visual_guided_attention(audio, visual, visual, mask)
        
        return visual_output, audio_output

class CLMP(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_hidden_dim, num_block=1, dropout=0.1, max_len=600):
        super(CLMP, self).__init__()
        
        self.visual_cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.audio_cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        self.add_position_emb = AddPositionalEmbedding(max_len+1, embedding_dim)
        
        self.internal_fusion_blocks = nn.ModuleList()
        self.external_fution_blocks = nn.ModuleList()
        
        for _ in range(num_block):
            self.internal_fusion_blocks.append(TransformerEncoderBlock(embedding_dim, num_heads, ff_hidden_dim, dropout))
            self.external_fution_blocks.append(ExternalFusionBlock(embedding_dim, num_heads, ff_hidden_dim, dropout))
            
    def forward(self, visual_feature, audio_feature, mask):
        visual_cls_tokens = self.visual_cls_token.expand(visual_feature.size(0), -1, -1)
        audio_cls_tokens = self.audio_cls_token.expand(audio_feature.size(0), -1, -1)
        
        visual_feature = torch.cat((visual_cls_tokens, visual_feature), dim=1)
        audio_feature = torch.cat((audio_cls_tokens, audio_feature), dim=1)
        mask = torch.cat((torch.zeros(visual_feature.size(0), 1, dtype=torch.bool, device=visual_feature.device), mask), dim=1)

        visual_feature = self.add_position_emb(visual_feature)
        audio_feature = self.add_position_emb(audio_feature)
        
        for internal_fusion_block in self.internal_fusion_blocks:
            visual_feature = internal_fusion_block(visual_feature, mask)
            audio_feature = internal_fusion_block(audio_feature, mask)
            
        visual_cls_representation = visual_feature[:, 1, :]
        audio_cls_representation =  audio_feature[:, 1, :]   
        
        for external_fution_block in self.external_fution_blocks:
            visual_feature, audio_feature = external_fution_block(visual_feature, audio_feature, mask)
            
        
        va_fusion_cls_representation = (visual_feature[:, 1, :] + audio_feature[:, 1, :]) / 2
        
        return visual_cls_representation, audio_cls_representation, va_fusion_cls_representation
    
class CLMPWithClassifier(nn.Module):
    def __init__(self, clmp_model, num_classes):
        super(CLMPWithClassifier, self).__init__()
        self.clmp_model = clmp_model  # 預訓練模型
        self.classifier = nn.Linear(clmp_model.embedding_dim, num_classes)  # 分類頭

    def forward(self, visual_feature, audio_feature, mask):
        # 使用原始模型的forward方法
        visual_cls_representation, audio_cls_representation, va_fusion_cls_representation = self.clmp_model(visual_feature, audio_feature, mask)
        
        # 使用分類頭
        logits = self.classifier(va_fusion_cls_representation)
        return logits

# import numpy as np
# from torch.cuda.amp import autocast
# import argparse
# import torch.distributed as dist
# from torch.utils.data.distributed import DistributedSampler

# parser = argparse.ArgumentParser()
# parser.add_argument("--local_rank", type=int)
# args = parser.parse_args()

# dist.init_process_group(backend='nccl')
# dist.barrier()
# world_size = dist.get_world_size()

# embedding_dim = 512
# num_heads = 8
# ff_hidden_dim = 1024
# num_blocks = 4
# dropout = 0.1
# max_len = 600
# seq_len = 600
# batch_size = 37

# device = torch.device("cuda:0")

# # 創建模型實例
# model = CLMP(embedding_dim, num_heads, ff_hidden_dim, num_blocks, dropout, max_len)
# model.to(device)

# # 創建隨機數據作為輸入
# visual_input = torch.randn(batch_size, seq_len, embedding_dim).to(device)
# audio_input = torch.randn(batch_size, seq_len, embedding_dim).to(device)

# # 創建一個簡單的遮罩
# mask = np.where(np.all(visual_input.cpu().numpy() == 0, axis=2), True, False)
# mask = torch.from_numpy(mask).to(device)

# # 通過模型傳遞輸入
# with autocast():
#     visual_cls_representation, audio_cls_representation, va_fusion_cls_representation = model(visual_input, audio_input, mask)

# # 檢查輸出
# print("Visual CLS Representation:", visual_cls_representation.shape)
# print("Audio CLS Representation:", audio_cls_representation.shape)
# print("VA Fusion CLS Representation:", va_fusion_cls_representation.shape)
        
