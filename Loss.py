import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features1, features2):
        """
        計算InfoNCE損失
        :param features1: 第一個模態的特徵，形狀應為 (batch_size, feature_dim)
        :param features2: 第二個模態的特徵，形狀應為 (batch_size, feature_dim)
        :return: InfoNCE損失
        """
        # 標準化特徵向量
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)

        # 計算相似度矩陣
        similarity_matrix = torch.matmul(features1, features2.T) / self.temperature

        # 計算損失
        batch_size = features1.size(0)
        labels = torch.arange(batch_size).to(features1.device)
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss
    
class DescriptionContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, v_d_weight=1, a_d_weight=1, va_d_weight=1):
        super(DescriptionContrastiveLoss, self).__init__()
        
        self.v_d_weight = v_d_weight
        self.a_d_weight = a_d_weight
        self.va_d_weight = va_d_weight
        
        self.contrastiveloss = InfoNCELoss(temperature)
        
    def _calculate_bilateral_loss(self, feature1, feature2):
        """
        計算雙向對比損失。
        """
        return self.contrastiveloss(feature1, feature2) + self.contrastiveloss(feature2, feature1)
    
    def forward(self, visual_feature, audio_feature, va_fusion_feature, description_feature):
        losses = [
            self.v_d_weight * self._calculate_bilateral_loss(visual_feature, description_feature),
            self.a_d_weight * self._calculate_bilateral_loss(audio_feature, description_feature),
            self.va_d_weight * self._calculate_bilateral_loss(va_fusion_feature, description_feature)
        ]
        
        description_loss = sum(losses)
        return description_loss
    
class GenreContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, v_g_weight=1, a_g_weight=1, va_g_weight=1):
        super(GenreContrastiveLoss, self).__init__()
        
        self.v_g_weight = v_g_weight
        self.a_g_weight = a_g_weight
        self.va_g_weight = va_g_weight
        
        self.contrastiveloss = InfoNCELoss(temperature)
        
    def _calculate_bilateral_loss(self, feature1, feature2):
        """
        計算雙向對比損失。
        """
        return self.contrastiveloss(feature1, feature2) + self.contrastiveloss(feature2, feature1)
    
    def forward(self, visual_feature, audio_feature, va_fusion_feature, genres_feature):
        genres_global_feature = torch.mean(genres_feature, dim=1)
        losses = [
            self.v_g_weight * self._calculate_bilateral_loss(visual_feature, genres_global_feature),
            self.a_g_weight * self._calculate_bilateral_loss(audio_feature, genres_global_feature),
            self.va_g_weight * self._calculate_bilateral_loss(va_fusion_feature, genres_global_feature)
        ]
        
        genres_loss = sum(losses)
        return genres_loss

class DescriptionWithGenresContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1,  v_d_weight=1, a_d_weight=1, va_d_weight=1, v_g_weight=1, a_g_weight=1, va_g_weight=1):
        super(DescriptionWithGenresContrastiveLoss, self).__init__()
        
        self.description_contrastive_loss = DescriptionContrastiveLoss(temperature, v_d_weight, a_d_weight, va_d_weight)
        self.genres_contrastive_loss = GenreContrastiveLoss(temperature, v_g_weight, a_g_weight, va_g_weight)
        
    def forward(self, visual_feature, audio_feature, va_fusion_feature, description_feature, genres_feature):
        description_loss = self.description_contrastive_loss(visual_feature, audio_feature, va_fusion_feature, description_feature)
        genre_loss = self.genres_contrastive_loss(visual_feature, audio_feature, va_fusion_feature, genres_feature)
        
        return description_loss + genre_loss

# class MultiLabelInfoNCE(nn.Module):
#     def __init__(self, temperature=0.1):
#         super(MultiLabelInfoNCE, self).__init__()
#         self.temperature = temperature
        
#     def _convert_to_one_hot(self, input_tensor, unique_tensor):
#         """
#         轉換 (batch, label_num, dim) 形狀的tensor為 (batch, all_class_num) 的one-hot編碼tensor。

#         :param tensor_large: 形狀為 (batch, label_num, dim) 的Tensor。
#         :param tensor_small: 形狀為 (all_class_num, dim) 的Tensor。
#         :return: 形狀為 (batch, all_class_num) 的one-hot編碼Tensor。
#         """
#         one_hot_encoded = torch.zeros(input_tensor.size(0), unique_tensor.size(0), dtype=torch.bool, device='cuda:1')

#         for i in range(input_tensor.size(0)):
#             distances = torch.cdist(input_tensor[i], unique_tensor)
#             min_distances, indices = torch.min(distances, dim=1)
#             for idx in indices:
#                 if idx == 0:
#                     continue
#                 one_hot_encoded[i][idx] = 1

#         return one_hot_encoded

#     def forward(self, anchor_features, label_features):
#         """
#         計算loss
#         :param anchor_features: 第一個模態的特徵，形狀應為 (batch_size, feature_dim)
#         :param label_features: 第二個模態的特徵，形狀應為 (batch_size, label_num, feature_dim)
#         :return: loss
#         """
#         print(label_features)
#         unique_features = torch.unique(label_features.view(-1, label_features.size(-1)), dim=0) #(all_class_num, feature_dim)
#         print(unique_features)
#         onehotvector = self._convert_to_one_hot(label_features, unique_features) #((batch, all_class_num)

#         anchor_features = F.normalize(anchor_features, dim=1)
#         unique_features = F.normalize(unique_features, dim=1)
        
#         similarity_matrix = torch.matmul(anchor_features, unique_features.T) / self.temperature #(batch_size, all_class_num)
#         pos_similarity = torch.sum(torch.exp(similarity_matrix) * onehotvector, dim=1) / torch.sum(onehotvector, dim=1)
#         denominator = torch.sum(torch.exp(similarity_matrix), dim=1)
        
#         loss = -torch.log(pos_similarity / denominator)

#         return loss.mean()
    
    
# class GenresFeatureContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.1, v_g_weight=1, a_g_weight=1, va_g_weight=1):
#         super(GenresFeatureContrastiveLoss, self).__init__()
        
#         self.v_g_weight = v_g_weight
#         self.a_g_weight = a_g_weight
#         self.va_g_weight = va_g_weight
        
#         self.multilabel_contrastiveloss = MultiLabelInfoNCE(temperature)
    
#     def forward(self, visual_feature, audio_feature, va_fusion_feature, genres_feature):
        
#         losses = [
#             self.v_g_weight * self.multilabel_contrastiveloss(visual_feature,  genres_feature),
#             self.a_g_weight * self.multilabel_contrastiveloss(audio_feature,  genres_feature),
#             self.va_g_weight * self.multilabel_contrastiveloss(va_fusion_feature,  genres_feature)
#         ]
        
#         genres_loss = sum(losses)
#         return genres_loss
    

# class GenresLabelContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.1):
#         super(GenresLabelContrastiveLoss, self).__init__()
        
#         self.learnable_weight = nn.Parameter(torch.tensor(1.0))
        
#     def _labels_to_one_hot(self, labels: List[List[int]], num_classes):
#         """
#         將multi label 轉換成 one-hot vector
        
#         :param labels: genres label list
#         :param num_classes: 總類別數
#         :return: one-hot vector
#         """
#         batch_size = len(labels)

#         one_hot_tensor = torch.zeros(batch_size, num_classes)

#         for i, label_list in enumerate(labels):
#             one_hot_tensor[i, label_list] = 1

#         return one_hot_tensor
        
#     def _jaccard_similarity_tensor(self, binary_matrix):
#         # 計算交集,聯集
#         intersection = torch.mm(binary_matrix, binary_matrix.t())
#         total = binary_matrix.sum(dim=1).unsqueeze(1)
#         union = total + total.t() - intersection

#         # 計算 Jaccard 
#         jaccard = intersection / union
#         jaccard.fill_diagonal_(0)
        
#         return jaccard
    
#     def _extract_elements(self, tensor, index):
#         result = []
#         for pair in index:
#             # Exclude the elements at the specified indices
#             indices_to_keep = [i for i in range(tensor.shape[1]) if i not in pair]
#             result.append(tensor[pair[0], indices_to_keep])
#         return torch.stack(result)
    
#     def forward(self, va_fusion_feature, genres_label):
#         # print(genres_label)
#         one_hot_label = self._labels_to_one_hot(genres_label, 30)
            
#         jaccard_sim = self._jaccard_similarity_tensor(one_hot_label).to(va_fusion_feature.device)
        
#         va_fusion_feature = F.normalize(va_fusion_feature, dim=1)
#         similarity_matrix = torch.matmul(va_fusion_feature, va_fusion_feature.T)
#         # similarity_matrix = similarity_matrix / (0.1 * jaccard_sim + 1e-10)
#         # print(similarity_matrix)
#         # similarity_matrix = self.sigmoid(similarity_matrix)
#         indices = torch.nonzero(jaccard_sim)
#         positive = similarity_matrix[indices[:,0], indices[:,1]].unsqueeze(1)
#         negative = self._extract_elements(similarity_matrix, indices)
#         logits = torch.cat([positive,negative], dim=1)
#         labels = torch.zeros(logits.size(0), dtype=torch.long).to(va_fusion_feature.device)
#         # print(logits, labels)
#         loss = F.cross_entropy(logits / 0.1, labels)
        
#         # genres_loss = F.binary_cross_entropy(similarity_matrix, jaccard_sim.to(similarity_matrix.device))

#         return loss
    

