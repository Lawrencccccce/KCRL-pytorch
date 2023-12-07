import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from .encoder import GATEncoder
from .decoder import SingleLayerDecoder


class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        self.config = config
        self.is_train = True
        # Data config
        self.batch_size = config.batch_size                     # batch size
        self.max_length = config.max_length                     # this is the number of nodes in the graph  
        self.input_dimension = config.input_dimension           # input dimension  

        # Reward config
        # self.avg_baseline = nn.Parameter(torch.Tensor([config.init_baseline]), requires_grad=False)
        # self.alpha = config.alpha  # moving average update

        # # Training config (actor)
        self.encoder = GATEncoder(config, is_train=self.is_train)
        # self.global_step = nn.Parameter(torch.Tensor([0]), requires_grad=False)  # global step
        # self.lr1_start = config.lr1_start  # initial learning rate
        # self.lr1_decay_rate = config.lr1_decay_rate  # learning rate decay rate
        # self.lr1_decay_step = config.lr1_decay_step  # learning rate decay step

        # # Training config (critic)
        self.decoder = SingleLayerDecoder(config, is_train=self.is_train)
        # self.global_step2 = nn.Parameter(torch.Tensor([0]), requires_grad=False)  # global step
        # self.lr2_start = config.lr1_start  # initial learning rate
        # self.lr2_decay_rate = config.lr1_decay_rate  # learning rate decay rate
        # self.lr2_decay_step = config.lr1_decay_step  # learning rate decay step

        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        self.input_ = torch.zeros((self.batch_size, self.max_length, self.input_dimension))
        self.reward_ = torch.zeros(self.batch_size)
        self.graphs_ = torch.zeros((self.batch_size, self.max_length, self.max_length))

    def forward(self, inputs):
        """
        input shape: (batch_size, max_length, input_dimension)
        output shape: (batch_size, max_length, input_embed) 
        """
        encoder_output = self.encoder.forward(inputs)
        decoder_output = self.decoder.forward(encoder_output)
        return decoder_output