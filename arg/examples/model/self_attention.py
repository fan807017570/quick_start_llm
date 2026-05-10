import torch
from sympy.polys.polyconfig import query

inputs = torch.tensor([
    [0.43, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.10],
    [0.05, 0.80, 0.55],
])

query = inputs[1]
attn_socre2 = torch.empty(inputs.shape[0])
# print(inputs.shape[0]) 6
# print(inputs.shape[1]) 3

for i, x_i in enumerate(inputs):
    attn_socre2[i] = torch.dot(x_i, query)
attn_weights_2_tmp = attn_socre2 / attn_socre2.sum()  # normization
print(attn_weights_2_tmp)
print(attn_weights_2_tmp.sum())


def softmax_naive(x):
    # x = [0.3,0.4,0.3]
    return torch.exp(x) / torch.exp(x).sum(dim=0)


attn_weights_2_tmp = softmax_naive(attn_socre2)
print(attn_weights_2_tmp)
print(attn_weights_2_tmp.sum())

query = input[1]
context_vec2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec2 += attn_weights_2_tmp[i] * x_i
print(context_vec2)
