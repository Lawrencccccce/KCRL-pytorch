import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    '''
        input shape: (batch_size, max_length, hidden_dim)
        output shape: (batch_size, 1)
    '''
    def __init__(self, config):
        super(Critic, self).__init__()
        self.config = config

        # Data config
        self.batch_size = config.batch_size 
        self.max_length = config.max_length 
        self.num_random_sample = config.num_random_sample 

        # Network config
        self.input_embed = config.hidden_dim 
        self.hidden_dim = config.hidden_dim 

        # Baseline setup
        self.init_baseline = 0.

        # Define layers
        self.dense = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.w1 = nn.Parameter(torch.Tensor(self.hidden_dim, 1))
        self.b1 = nn.Parameter(torch.Tensor(1))
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.xavier_uniform_(self.w1)
        self.b1.data.fill_(self.init_baseline)

    def forward(self, encoder_output):
        # [Batch size, max_length, hidden_dim] to [Batch size]
        
        frame = torch.mean(encoder_output, dim=1)                           # shape (batch_size, hidden_dim)
        

        # ffn 1
        h0 = F.relu(self.dense(frame))                                      # shape (batch_size, hidden_dim)
        # ffn 2
        self.predictions = torch.squeeze(torch.mm(h0, self.w1) + self.b1)   # shape (batch_size)

        return self.predictions