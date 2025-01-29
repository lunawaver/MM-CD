import torch.nn as nn

class Multihead_Attention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(Multihead_Attention, self).__init__()
        self.feature_dim = feature_dim

        self.num_heads = num_heads

        self.attention_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads)

        self.attention_model = nn.TransformerEncoder(self.attention_layer, num_layers=8)
        
    def forward(self, x):
        attn_output = self.attention_model(x)

        return attn_output

class Channel_Attention(nn.Module):
    def __init__(self, channel_dim, num_heads):
        super(Channel_Attention, self).__init__()
        self.channel_dim = channel_dim

        self.num_heads = num_heads

        self.channel_attention_model = nn.Transformer(d_model=channel_dim, nhead=num_heads, num_encoder_layers=1, num_decoder_layers=1, batch_first=True)

    def forward(self, query, key, value):
        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)
        value = value.permute(2, 0, 1)
        
        channel_attn_output = self.channel_attention_model(value, query).permute(1, 2, 0)

        return channel_attn_output

class Spiral_Attention(nn.Module):
    def __init__(self, spiral_dim, num_heads):
        super(Spiral_Attention, self).__init__()
        self.spiral_dim = spiral_dim

        self.num_heads = num_heads
 
        self.spiral_attention_model = nn.Transformer(d_model=spiral_dim, nhead=num_heads, num_encoder_layers=1, num_decoder_layers=1, batch_first=True)

    def forward(self, query, key, value):
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        spiral_attn_output = self.spiral_attention_model(value, query).permute(1, 0, 2)

        return spiral_attn_output
