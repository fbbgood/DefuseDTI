import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from torch.nn import MultiheadAttention
from typing import Optional, Tuple


class MultiModal_Attention_mechanism(nn.Module):
    def __init__(self):
        super().__init__()

        self.hiddim = 3
        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_x1 = nn.Linear(in_features=3, out_features=self.hiddim)
        self.fc_x2 = nn.Linear(in_features=self.hiddim, out_features=3)
        self.sigmoidx = nn.Sigmoid()

    def forward(self,input_list):
        new_input_list1 = input_list[0].reshape(1, 1, input_list[0].shape[0], -1)
        new_input_list2 = input_list[1].reshape(1, 1, input_list[1].shape[0], -1)
        new_input_list3 = input_list[2].reshape(1, 1, input_list[2].shape[0], -1)
        XM = torch.cat((new_input_list1, new_input_list2, new_input_list3), 1)

        x_channel_attenttion = self.globalAvgPool(XM)

        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), -1)
        x_channel_attenttion = self.fc_x1(x_channel_attenttion)
        x_channel_attenttion = torch.relu(x_channel_attenttion)
        x_channel_attenttion = self.fc_x2(x_channel_attenttion)
        x_channel_attenttion = self.sigmoidx(x_channel_attenttion)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), x_channel_attenttion.size(1), 1, 1)

        XM_channel_attention = x_channel_attenttion * XM
        XM_channel_attention = torch.relu(XM_channel_attention)

        return XM_channel_attention[0]

class MFA(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, num_heads=4, dropout=0.1, pooling_k=3):
        super(MFA, self).__init__()

        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out
        self.num_heads = num_heads
        self.pooling_k = pooling_k

        self.v_net = FCNet([v_dim, h_dim], dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim], dropout=dropout)

        self.multihead_attention = MultiheadAttention(h_dim, num_heads, dropout=dropout)

        self.output_net = FCNet([h_dim, h_out], dropout=dropout)

        # Add Pooling layer
        if 1 < pooling_k:
            self.pooling = nn.AvgPool1d(pooling_k, stride=pooling_k)
        else:
            self.pooling = None

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


