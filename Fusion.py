import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from torch.nn import MultiheadAttention
from typing import Optional, Tuple



    def forward(self, v, q):
        v = self.v_net(v)
        q = self.q_net(q)

        # MultiheadAttention requires input shape (sequence_length, h_dim)
        v = v.transpose(0, 1)
        q = q.transpose(0, 1)

        attn_output, attn_weights = self.multihead_attention(q, v, v)

        # Transpose back to original shape (h_dim, sequence_length, )
        attn_output = attn_output.transpose(0, 1)
        attn_weights = attn_weights.transpose(0, 1)
        # Apply Pooling
        if self.pooling is not None:
            attn_output = attn_output.permute(0, 2, 1)
            attn_output = self.pooling(attn_output)
            attn_output = attn_output.permute(0, 2, 1)

        output = self.output_net(attn_output)

        return output, attn_weights

class FCNet(nn.Module):
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


