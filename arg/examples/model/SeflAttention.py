import torch
import torch.nn as nn
from transformers.data.processors.squad import squad_convert_example_to_features_init


class SeflAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        query = x @ self.W_query
        value = x @ self.W_value
        attn_scores = query @ keys.T
        context_len = attn_scores.shape[0]
        # mask_sample = torch.tril(torch.ones(context_len, context_len))
        mask_sample = torch.triu(torch.ones(context_len,context_len),diagonal=1)
        print(mask_sample)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        # masked_weights = attn_weights * mask_sample
        masked_weights = attn_weights.masked_fill(mask_sample.bool(),-torch.inf)
        print(masked_weights)

        context_vec = masked_weights @ value
        return context_vec;


def test():
    x1 = torch.tensor([0.1, 0.9, 0.7])
    w1 = torch.tensor([[0.1, 0.1, 0.2],
                       [0.2, 0.2, 0.5],
                       [0.6, 0.9, 0.7]
                       ])

    print(x1 @ w1)


def dot(x, y):
    return sum(x_i * y_i for x_i, y_i in zip(x, y))


def test1():
    x1 = torch.tensor([0.1, 0.9, 0.7])
    y1 = torch.tensor([0.1, 0.2, 0.6])
    print(dot(x1, y1))


if __name__ == "__main__":
    # torch.manual_seed(123)
    # sa_v1 = SeflAttention(d_in, d_out)
    # print(sa_v1(inputs))
    sa = SeflAttention(2, 2)

    inputs = torch.tensor([[0.9996, 0.8053],
                           [0.5061, 0.8210],
                           [0.3058, 0.8203],
                           [0.6948, 0.7939],
                           [0.9927, 0.7891],
                           [0.2990, 0.8040]])
    print(sa.forward(inputs))
