import torch.nn as nn
from config import config as cfg
from transformers import ConvNextV2Model

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.ConvNeXT = ConvNextV2Model.from_pretrained(cfg.ConvNeXT_NAME)
    def forward(self, images):
        ConvNeXT_output = self.ConvNeXT(images).last_hidden_state
        return ConvNeXT_output

