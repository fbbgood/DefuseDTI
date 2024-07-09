import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN,GAT
from torch.nn.utils.weight_norm import weight_norm
from Fusion import MFA



class DefuseDTI(nn.Module):
    def __init__(self, **config):
        super(DefuseDTI, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        self_attention_heads = config["SELF_ATTENTION"]["HEADS"]
        self_attention_h_dim = config["SELF_ATTENTION"]["HIDDEN_DIM"]
        self_attention_h_out = config["SELF_ATTENTION"]["OUT_DIM"]

        self.drug_extractor = DrugGraphNet(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                      padding=drug_padding,
                                      hidden_feats=drug_hidden_feats)

        self.protein_extractor_rs = ProteinRS(protein_emb_dim, num_filters, kernel_size, protein_padding)
        self.protein_extractor_ir = ProteinIR(protein_emb_dim, num_filters, kernel_size, protein_padding)

        self.self_attention = MFA(v_dim=drug_hidden_feats[-1], q_dim=protein_emb_dim, h_dim=self_attention_h_dim,
                                  h_out=self_attention_h_out, num_heads=self_attention_heads)
        mlp_in_dim = self_attention_h_out

        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, v_p, mode="train"):


        v_d = self.drug_extractor(bg_d)


        v_p_rs = self.protein_extractor_rs(v_p)
        v_p_ir = self.protein_extractor_ir(v_p)


        v_p = torch.cat((v_p_rs, v_p_ir), dim=1)   # Combine the features from both branches

        f, att = self.self_attention(v_d, v_p)
        f_pooled = torch.mean(f, dim=1)  # Global max pooling
        score = self.mlp_classifier(f_pooled)

        if mode == "train":
            return v_d, v_p, f, score

        elif mode == "eval":
            return v_d, v_p, score, att


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.match_dim = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv3(out)
        out = self.bn3(out)

        # 计算匹配维度卷积层的输出，以便输入和输出的尺寸匹配
        if x.shape[-1] != out.shape[-1]:
            residual = F.pad(residual, (0, out.shape[-1] - x.shape[-1]))

        residual = self.match_dim(residual)
        out += residual
        out = self.relu(out)
        return out

class ProteinRS(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinRS, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.res_block1 = ResidualBlock(in_ch[0], in_ch[1], kernels[0])
        self.res_block2 = ResidualBlock(in_ch[1], in_ch[2], kernels[1])


    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.res_block1(v)
        v = self.res_block2(v)
        v = v.view(v.size(0), v.size(2), -1)
        return v

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss

def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss

def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent
