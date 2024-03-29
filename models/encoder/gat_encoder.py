import torch
import torch.nn as nn
import torch.nn.functional as F

def attn_head(seq, out_sz, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    '''
    input shape: (batch_size, max_length, input_dimension)
    output shape: (batch_size, max_length, out_sz)
    '''
    if in_drop != 0.0:
        seq = F.dropout(seq, in_drop)                               # shape (batch_size, max_length, input_dimension)

    seq = seq.permute(0, 2, 1)                                      # shape (batch_size, input_dimension, max_length)
    # print(f'seq shape: {seq.shape}')
    seq_fts = nn.Conv1d(seq.size(1), out_sz, 1, bias=False)(seq)    # shape (batch_size, out_sz, max_length)

    # print(f'seq_fts shape: {seq_fts.shape}')

    f_1 = nn.Conv1d(out_sz, 1, 1)(seq_fts)                          # shape (batch_size, 1, max_length)   
    f_2 = nn.Conv1d(out_sz, 1, 1)(seq_fts)                          # shape (batch_size, 1, max_length)
    # print(f'f_1 shape: {f_1.shape}')
    # print(f'f_2 shape: {f_2.shape}')
    logits = f_1.permute(0, 2, 1) + f_2                             # shape (batch_size, max_length, max_length)
    # print(f'logits shape: {logits.shape}')
    coefs = F.softmax(F.leaky_relu(logits), dim=-1)                 

    seq_fts = seq_fts.permute(0, 2, 1)                              # shape (batch_size, max_length, out_sz)
    if coef_drop != 0.0:
        coefs = F.dropout(coefs, coef_drop)                         # shape (batch_size, max_length, max_length)
    if in_drop != 0.0:
        seq_fts = F.dropout(seq_fts, in_drop)                       # shape (batch_size, max_length, out_sz)

    # print(f'coefs shape: {coefs.shape}')
    # print(f'seq_fts shape: {seq_fts.shape}')

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
        """
        input shape: (batch_size, max_length, input_dimension)
        output shape: (batch_size, max_length, input_embed) 
        """
        head_hidden_dim = int(self.hidden_dim / self.num_heads)
        # print(f'head_hidden_dim: {head_hidden_dim}')
        x = inputs
        for _ in range(self.num_stacks):
            heads = []
            for _ in range(self.num_heads):
                head = attn_head(x, head_hidden_dim, activation=F.elu, in_drop=0.0, coef_drop=0.0, residual=self.residual)
                heads.append(head)
            x = torch.cat(heads, dim=-1)
        # logits = nn.Linear(self.hidden_dim * self.num_heads, self.input_dimension)(x)
        return x