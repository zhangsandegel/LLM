import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 位置编码（Positional Encoding）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建一个长 max_len 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x的shape是 (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

# Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model
    
    def forward(self, query, key, value, mask=None):
        # query, key, value 的形状都是 (batch_size, num_heads, seq_len, d_model // num_heads)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_model)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        # 定义 Q, K, V 的线性层
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 对 Q, K, V 进行线性变换并划分成多个头
        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.d_k)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.d_k)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.d_k)
        
        query = query.transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # 计算注意力输出
        output, attention_weights = self.attention(query, key, value, mask)
        
        # 将所有头的输出拼接起来
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 通过输出线性层
        output = self.out_linear(output)
        return output, attention_weights

# Position-wise Feed-Forward Networks (FFN)
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self Attention
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        # Feed-Forward
        ff_output = self.ffn(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        
        return x

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, mask=None, enc_mask=None):
        # Self Attention
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        # Cross Attention (与Encoder输出交互)
        cross_attn_output, _ = self.cross_attention(x, enc_output, enc_output, enc_mask)
        x = self.layer_norm2(x + self.dropout(cross_attn_output))
        
        # Feed-Forward
        ff_output = self.ffn(x)
        x = self.layer_norm3(x + self.dropout(ff_output))
        
        return x

# Transformer模型
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, num_classes, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder 和 Decoder的层
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.fc_out = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.embedding(src)
        tgt = self.embedding(tgt)

        src = self.pos_encoding(src)
        tgt = self.pos_encoding(tgt)

        # Encoder
        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        # Decoder
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, tgt_mask, src_mask)

        output = self.fc_out(tgt)
        return output

# Example usage:
# 假设我们有词汇大小5000，输入序列长度10，目标类别数为2
model = Transformer(vocab_size=5000, d_model=512, num_heads=8, d_ff=2048, num_layers=6, num_classes=2)

# 随机输入
src = torch.randint(0, 5000, (32, 10))  # batch_size = 32, seq_len = 10
tgt = torch.randint(0, 5000, (32, 10))  # batch_size = 32, seq_len = 10

output = model(src, tgt)
print(output.shape)  # 输出形状: (batch_size, seq_len, num_classes)
