import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import GATEncoder
from .decoder import SingleLayerDecoder


class Actor(nn.Module):
    '''
        input: inputs (batch_size, max_length, num_random_sample)
        output: samples (batch_size, max_length, max_length)
    '''


    def __init__(self, config):
        super(Actor, self).__init__()
        self.config = config
        self.is_train = True
        # Data config
        self.batch_size = config.batch_size                         # batch size
        self.max_length = config.max_length                         # this is the number of nodes in the graph  
        self.num_random_sample = config.num_random_sample           # input dimension 

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
        self.input_ = torch.zeros((self.batch_size, self.max_length, self.num_random_sample))
        self.reward_ = torch.zeros(self.batch_size)
        self.graphs_ = torch.zeros((self.batch_size, self.max_length, self.max_length))

    def forward(self, inputs):
        """
            input shape: (batch_size, max_length, num_random_sample)
            output shape: (batch_size, max_length, max_length)
        """
        encoder_output = self.encoder.forward(inputs)
        samples, mask_scores, entropy = self.decoder.forward(encoder_output)
        graph_gen = torch.transpose(torch.stack(samples), 0, 1)
        logits_for_rewards = torch.stack(mask_scores).permute(1, 0, 2)
        entropy_for_rewards = torch.stack(entropy).permute(1, 0, 2)

        test_scores = torch.sigmoid(logits_for_rewards)[:2]
        log_probss = F.binary_cross_entropy_with_logits(logits_for_rewards, graph_gen, reduction='none')
        log_softmax = torch.mean(log_probss, axis=[1, 2])
        entropy_regularization = torch.mean(entropy_for_rewards, axis=[1,2])
        return encoder_output, graph_gen, log_softmax, entropy_regularization