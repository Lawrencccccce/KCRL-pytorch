import torch
import torch.nn as nn
import torch.nn.functional as F



class GATEncoder(nn.Module):

    '''
        input shape: (batch_size, max_length, num_random_sample)
        output shape: (batch_size, max_length, hidden_dim)
    '''
    def __init__(self, config, is_train):
        super(GATEncoder, self).__init__()
        self.config = config
        self.max_length = config.max_length                 # this is the number of nodes in the graph 

        self.batch_size = config.batch_size                 # batch size
        self.num_random_sample = config.num_random_sample       # input dimension   

        self.hidden_dim = config.hidden_dim                 # hidden dimension
        self.num_heads = config.num_heads                   # number of heads
        self.num_stacks = config.num_stacks                 # number of stacks
        self.residual = config.residual                     # residual connection
        self.is_training = is_train


    def forward(self, inputs):
        """
        input shape: (batch_size, max_length, num_random_sample) d by n
        output shape: (batch_size, max_length, hidden_dim)      
        """

        # hidden_dim must be divided by the num_heads
        head_hidden_dim = int(self.hidden_dim / self.num_heads)

        x = inputs
        for _ in range(self.num_stacks):
            heads = []
            for _ in range(self.num_heads):
                head = self.attn_head(x, head_hidden_dim, activation=F.elu, in_drop=0.0, coef_drop=0.0, residual=self.residual)
                heads.append(head)
            x = torch.cat(heads, dim=-1)
        # logits = nn.Linear(self.hidden_dim * self.num_heads, self.num_random_sample)(x)
        return x
    

    def attn_head(self, seq, out_sz, activation, in_drop=0.0, coef_drop=0.0, residual=False):
        '''
        input shape: (batch_size, max_length, num_random_sample)
        output shape: (batch_size, max_length, out_sz)
        '''
        if in_drop != 0.0:
            seq = F.dropout(seq, in_drop)                               # shape (batch_size, max_length, num_random_sample)

        
        seq = seq.permute(0, 2, 1)                                      # shape (batch_size, num_random_sample, max_length)
        seq_fts = nn.Conv1d(seq.size(1), out_sz, 1, bias=False)(seq)    # shape (batch_size, out_sz, max_length)

        f_1 = nn.Conv1d(out_sz, 1, 1)(seq_fts)                          # shape (batch_size, 1, max_length)   
        f_2 = nn.Conv1d(out_sz, 1, 1)(seq_fts)                          # shape (batch_size, 1, max_length)
        
        logits = f_1.permute(0, 2, 1) + f_2                             # shape (batch_size, max_length, max_length)
        
        coefs = F.softmax(F.leaky_relu(logits), dim=-1)                 

        seq_fts = seq_fts.permute(0, 2, 1)                              # shape (batch_size, max_length, out_sz)
        if coef_drop != 0.0:
            coefs = F.dropout(coefs, coef_drop)                         # shape (batch_size, max_length, max_length)
        if in_drop != 0.0:
            seq_fts = F.dropout(seq_fts, in_drop)                       # shape (batch_size, max_length, out_sz)
        
        vals = torch.bmm(coefs, seq_fts)
        ret = vals + seq_fts

        if residual:
            if seq.size(-1) != ret.size(-1):
                ret = ret + nn.Conv1d(seq.size(1), ret.size(-1), 1)(seq)
            else:
                ret = ret + seq

        return activation(ret)
