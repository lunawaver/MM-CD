import torch.nn as nn

class MultiImageWeightGenerator(nn.Module):
    def __init__(self, seq_len=49, feature_dim=768, num_heads=12):
        super(MultiImageWeightGenerator, self).__init__()
        self.multihead_attention_layer1 = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads)
        self.norm_layer1 = nn.BatchNorm1d(seq_len)
        self.weight_generator_layer1 = nn.Linear(feature_dim * seq_len, 1)

        self.multihead_attention_layer2 = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads)
        self.norm_layer2 = nn.BatchNorm1d(seq_len * 2)
        self.weight_generator_layer2 = nn.Linear(feature_dim * seq_len * 2, 1)

        self.multihead_attention_layer3 = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads)
        self.norm_layer3 = nn.BatchNorm1d(seq_len * 4)
        self.weight_generator_layer3 = nn.Linear(feature_dim * seq_len * 4, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        slice_num, seq_len, feature_dim = x.shape
        x_temp = x
        x_temp = self.multihead_attention_layer1(x_temp)
        x_temp = self.norm_layer1(x_temp)
        x = x + x_temp
        x = x.flatten(1)
        weight1 = self.sigmoid(self.weight_generator_layer1(x))

        x = x.view(int(slice_num / 2), seq_len * 2, feature_dim)
        x_temp = x
        x_temp = self.multihead_attention_layer2(x_temp)
        x_temp = self.norm_layer2(x_temp)
        x = x + x_temp
        x = x.view(int(slice_num / 2), -1)
        weight2 = self.sigmoid(self.weight_generator_layer2(x)) / 2
        weight2 = weight2.repeat_interleave(2, dim=0)

        x = x.view(int(slice_num / 4), seq_len * 4, feature_dim)
        x_temp = x
        x_temp = self.multihead_attention_layer3(x_temp)
        x_temp = self.norm_layer3(x_temp)
        x = x + x_temp
        x = x.view(int(slice_num / 4), -1)
        weight3 = self.sigmoid(self.weight_generator_layer3(x)) / 4
        weight3 = weight3.repeat_interleave(4, dim=0)

        weight1 = weight1 + weight2
        weight1 = weight1 + weight3
        return weight1
