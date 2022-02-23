from torch import nn
from fastNLP.modules import LSTM
import torch

# 定义模型
class BiLSTMMaxPoolCls(nn.Module):
    def __init__(self, embed, num_classes, hidden_size=100, num_layers=5, dropout=0.3):
        super().__init__()
        self.embed = embed
        
        self.lstm = LSTM(self.embed.embedding_dim, hidden_size=hidden_size//2, num_layers=num_layers, 
                         batch_first=True, bidirectional=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, chars, seq_len):  # 这里的名称必须和DataSet中相应的field对应，比如之前我们DataSet中有chars，这里就必须为chars
        # chars:[batch_size, max_len]
        # seq_len: [batch_size, ]
        # print("chars1:",type(chars), chars.size())
        chars = self.embed(chars)
        # print("chars2:",type(chars), chars.size())
        outputs, _ = self.lstm(chars, seq_len)
        # print("outputs1:",outputs.size())
        outputs = self.dropout_layer(outputs)
        # print("outputs2:",outputs.size())
        outputs, _ = torch.max(outputs, dim=1)
        # print("outputs3:",outputs.size())
        outputs = self.fc(outputs)
        # print("outputs4:",outputs.size(),outputs)
        
        return {'pred':outputs}  # [batch_size,], 返回值必须是dict类型，且预测值的key建议设为pred

# 初始化模型
# model = BiLSTMMaxPoolCls(word2vec_embed, len(data_bundle.get_vocab('target')))