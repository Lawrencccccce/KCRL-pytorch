import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        self.config = config

        # Data config
        self.batch_size = config.batch_size 
        self.max_length = config.max_length 
        self.input_dimension = config.input_dimension 

        # Network config
        self.input_embed = config.hidden_dim 
        self.num_neurons = config.hidden_dim 

        # Baseline setup
        self.init_baseline = 0.

        # Define layers
        self.dense = nn.Linear(self.num_neurons, self.num_neurons)
        self.w1 = nn.Parameter(torch.Tensor(self.num_neurons, 1))
        self.b1 = nn.Parameter(torch.Tensor(1))
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.xavier_uniform_(self.w1)
        self.b1.data.fill_(self.init_baseline)

    def forward(self, encoder_output):
        # [Batch size, Sequence Length, Num_neurons] to [Batch size, Num_neurons]
        frame = torch.mean(encoder_output, dim=1) 

        # ffn 1
        h0 = F.relu(self.dense(frame))
        # ffn 2
        self.predictions = torch.squeeze(torch.mm(h0, self.w1) + self.b1)

        return self.predictions