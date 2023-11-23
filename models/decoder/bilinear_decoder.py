import torch
import torch.nn as nn

class BilinearDecoder(nn.Module):
    def __init__(self, config, is_train):
        super(BilinearDecoder, self).__init__()
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.input_dimension = config.hidden_dim
        self.input_embed = config.hidden_dim
        self.use_bias = config.use_bias
        self.bias_initial_value = config.bias_initial_value
        self.use_bias_constant = config.use_bias_constant
        self.is_training = is_train

        self.samples = []
        self.mask = 0
        self.mask_scores = []
        self.entropy = []

        self.W = nn.Parameter(torch.Tensor(self.input_embed, self.input_embed))
        nn.init.xavier_uniform_(self.W)

        if self.bias_initial_value is None:
            self.logit_bias = nn.Parameter(torch.Tensor(1))
        elif self.use_bias_constant:
            self.logit_bias = nn.Parameter(torch.Tensor([self.bias_initial_value]))
        else:
            self.logit_bias = nn.Parameter(torch.Tensor([self.bias_initial_value]))


    def decode(self, encoder_output):
        logits = torch.einsum('ijk,kl,ilm->ijm', encoder_output, self.W, encoder_output)
        if self.use_bias:
            logits += self.logit_bias

        self.adj_prob = logits

        for i in range(self.max_length):
            position = torch.ones(encoder_output.shape[0], dtype=torch.int32) * i
            self.mask = torch.nn.functional.one_hot(position, self.max_length)
            masked_score = self.adj_prob[:,i,:] - 100000000. * self.mask

        return masked_score