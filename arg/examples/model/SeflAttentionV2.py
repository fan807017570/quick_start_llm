import torch
from torch import nn


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # 返回一个可被学校的张量， bias用于设置是否需要偏置
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_len, context_len), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        print(attn_scores.shape)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens,:num_tokens], -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        contextvec = attn_weights @ values
        return contextvec


if __name__ == '__main__':
    torch.manual_seed(123)
    batch = torch.randn(2,4,3)
    context_len = batch.shape[1]
    print(batch)
    ca = CausalAttention(3, 3, context_len, 0.0)
    context_vec = ca.forward(batch)
    print("context_vecs.shape:", context_vec.shape)
    print(context_vec)
