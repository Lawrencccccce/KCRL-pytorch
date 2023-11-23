import torch
import torch.nn as nn
import torch.nn.functional as F

def attn_head(seq, out_sz, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    if in_drop != 0.0:
        seq = F.dropout(seq, in_drop)

    seq_fts = nn.Conv1d(seq.size(1), out_sz, 1, bias=False)(seq)

    f_1 = nn.Conv1d(out_sz, 1, 1)(seq_fts)
    f_2 = nn.Conv1d(out_sz, 1, 1)(seq_fts)
    logits = f_1 + f_2.permute(0, 2, 1)
    coefs = F.softmax(F.leaky_relu(logits), dim=-1)

    if coef_drop != 0.0:
        coefs = F.dropout(coefs, coef_drop)
    if in_drop != 0.0:
        seq_fts = F.dropout(seq_fts, in_drop)

    vals = torch.bmm(coefs, seq_fts)
    ret = vals + seq_fts

    if residual:
        if seq.size(-1) != ret.size(-1):
            ret = ret + nn.Conv1d(seq.size(1), ret.size(-1), 1)(seq)
        else:
            ret = ret + seq

    return activation(ret)


class GATEncoder(nn.Module):
    def __init__(self, config, is_train):
        super(GATEncoder, self).__init__()
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.input_dimension = config.input_dimension
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_stacks = config.num_stacks
        self.residual = config.residual
        self.is_training = is_train

    def encode(self, inputs):
        x = inputs
        for _ in range(self.num_stacks):
            heads = []
            for _ in range(self.num_heads):
                head = attn_head(x, self.hidden_dim, activation=F.elu, in_drop=0.0, coef_drop=0.0, residual=self.residual)
                heads.append(head)
            x = torch.cat(heads, dim=-1)
        logits = nn.Linear(self.hidden_dim * self.num_heads, self.input_dimension)(x)
        return logits