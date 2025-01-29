import torch.nn as nn

from model.MultiheadAttention import Multihead_Attention

class MultiModelFusionModel(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout_p=0.0):
        super(MultiModelFusionModel, self).__init__()
        self.multihead_attention_layers = Multihead_Attention(feature_dim=feature_dim, num_heads=num_heads, dropout_p=dropout_p)

    def forward(self, x):
        x = self.multihead_attention_layers(x)
        return x
