import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MultiheadAttention import Channel_Attention, Spiral_Attention

class MultiImageFusionModel(nn.Module):
    def __init__(self):
        super(MultiImageFusionModel, self).__init__()
        self.channel_attention_layer1 = Channel_Attention(768, 12)
        self.spiral_attention_layer1 = Spiral_Attention(49, 7)
        self.encoder1 = nn.Conv1d(80, 8, kernel_size=1, bias=False)
        self.Normalize_layer1 = torch.nn.BatchNorm1d(8)

        self.channel_attention_layer2 = Channel_Attention(768, 12)
        self.spiral_attention_layer2 = Spiral_Attention(49, 7)
        self.encoder2 = nn.Conv1d(16, 4, kernel_size=1, bias=False)
        self.Normalize_layer2 = torch.nn.BatchNorm1d(4)

        self.channel_attention_layer3 = Channel_Attention(768, 12)
        self.spiral_attention_layer3 = Spiral_Attention(49, 7)
        self.decoder1 = nn.Conv1d(16, 4, kernel_size=1, bias=False)
        self.Normalize_layer3 = torch.nn.BatchNorm1d(4)

        self.channel_attention_layer4 = Channel_Attention(768, 12)
        self.spiral_attention_layer4 = Spiral_Attention(49, 7)
        self.decoder2 = nn.Conv1d(48, 8, kernel_size=1, bias=False)
        self.Normalize_layer4 = torch.nn.BatchNorm1d(8)

        self.projection_layer = torch.nn.Conv1d(8, 1, kernel_size=1, padding=0, bias=False)

    def forward(self, selected_image_features):
        selected_image_features = selected_image_features.flatten(2)

        channel_layer1_input = selected_image_features
        spiral_layer1_input = selected_image_features
        channel_layer1_output = self.channel_attention_layer1(channel_layer1_input, channel_layer1_input, channel_layer1_input)
        spiral_layer1_output = self.spiral_attention_layer1(spiral_layer1_input, spiral_layer1_input, spiral_layer1_input)
        attention_layer1_output = torch.cat((channel_layer1_output, spiral_layer1_output), dim=0).permute(2, 0, 1)
        encoder1_output = (self.Normalize_layer1(F.silu(self.encoder1(attention_layer1_output)))).permute(1, 2, 0)

        channel_layer2_input = encoder1_output
        spiral_layer2_input = encoder1_output
        channel_layer2_output = self.channel_attention_layer2(channel_layer2_input, channel_layer2_input, channel_layer2_input)
        spiral_layer2_output = self.spiral_attention_layer2(spiral_layer2_input, spiral_layer2_input, spiral_layer2_input)
        attention_layer2_output = torch.cat((channel_layer2_output, spiral_layer2_output), dim=0).permute(2, 0, 1)
        encoder2_output = self.Normalize_layer2(F.silu(self.encoder2(attention_layer2_output))).permute(1, 2, 0)

        channel_layer3_input = encoder2_output
        spiral_layer3_input = encoder2_output
        channel_layer3_output = self.channel_attention_layer3(channel_layer3_input, channel_layer3_input, channel_layer3_input)
        spiral_layer3_output = self.spiral_attention_layer3(spiral_layer3_input, spiral_layer3_input, spiral_layer3_input)
        attention_layer3_output = torch.cat((channel_layer3_output, spiral_layer3_output, encoder1_output), dim=0).permute(2, 0, 1)
        decoder1_output = self.Normalize_layer3(F.silu(self.decoder1(attention_layer3_output))).permute(1, 2, 0)

        channel_layer4_input = decoder1_output
        spiral_layer4_input = decoder1_output
        channel_layer4_output = self.channel_attention_layer4(channel_layer4_input, channel_layer4_input, channel_layer4_input)
        spiral_layer4_output = self.spiral_attention_layer4(spiral_layer4_input, spiral_layer4_input, spiral_layer4_input)
        attention_layer4_output = torch.cat((channel_layer4_output, spiral_layer4_output, selected_image_features), dim=0).permute(2, 0, 1)
        decoder2_output = self.Normalize_layer4(F.silu(self.decoder2(attention_layer4_output))).permute(1, 2, 0)

        projection_layer_input = decoder2_output.permute(2, 0, 1)
        image_fusion_output = self.projection_layer(projection_layer_input).permute(1, 0, 2)

        return image_fusion_output
