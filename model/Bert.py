import torch.nn as nn
from transformers import BertConfig, BertModel
import config.config as cfg

class ReportEncoder(nn.Module):
    def __init__(self):
        super(ReportEncoder, self).__init__()
        self.bert_config = BertConfig.from_pretrained(cfg.Bert_NAME, output_hidden_states=True, output_attentions=True)
        self.bert = BertModel.from_pretrained(cfg.Bert_NAME).to("cuda")

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).last_hidden_state.to("cuda")
