import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleLayerDecoder(object):

    def __init__(self, config, is_train):
        super(SingleLayerDecoder, self).__init__()
        self.batch_size = config.batch_size    # batch size
        self.max_length = config.max_length    # input sequence length (number of cities)
        self.input_dimension = config.hidden_dim
        self.input_embed = config.hidden_dim    # dimension of embedding space (actor)
        self.decoder_hidden_dim = config.decoder_hidden_dim
        self.decoder_activation = config.decoder_activation
        self.use_bias = config.use_bias
        self.bias_initial_value = config.bias_initial_value
        self.use_bias_constant = config.use_bias_constant

        self.is_training = is_train

        self.samples = []
        self.mask = 0
        self.mask_scores = []
        self.entropy = []




    def forward(self, encoder_output):
        '''
            encoder_output is a tensor of size [batch_size, max_length, input_embed]
        '''
        W_l = nn.Parameter(torch.randn(self.input_embed, self.decoder_hidden_dim))
        W_r = nn.Parameter(torch.randn(self.input_embed, self.decoder_hidden_dim))
        U = nn.Parameter(torch.randn(self.decoder_hidden_dim))

        dot_l = torch.einsum('ijk, kl->ijl', encoder_output, W_l)                   # shape (batch_size, max_length, decoder_hidden_dim)
        dot_r = torch.einsum('ijk, kl->ijl', encoder_output, W_r)                   # shape (batch_size, max_length, decoder_hidden_dim)

        tiled_l = dot_l.unsqueeze(2).expand(-1, -1, self.max_length, -1)            # shape (batch_size, max_length, max_length, decoder_hidden_dim)
        tiled_r = dot_r.unsqueeze(1).expand(-1, self.max_length, -1, -1)            # shape (batch_size, max_length, max_length, decoder_hidden_dim)

        if self.decoder_activation == 'tanh':
            final_sum = torch.tanh(tiled_l + tiled_r)
        elif self.decoder_activation == 'relu':
            final_sum = F.relu(tiled_l + tiled_r)
        elif self.decoder_activation == 'none':
            final_sum = tiled_l + tiled_r
        else:
            raise NotImplementedError('Current decoder activation is not implemented yet')

        logits = torch.einsum('ijkl, l->ijk', final_sum, U)                         # shape (batch_size, max_length, max_length)

        if self.bias_initial_value is None:
            self.logit_bias = nn.Parameter(torch.randn(1))
        elif self.use_bias_constant:
            self.logit_bias = torch.tensor([self.bias_initial_value], dtype=torch.float32)
        else:
            self.logit_bias = nn.Parameter(torch.tensor([self.bias_initial_value], dtype=torch.float32))

        self.adj_prob = logits


        for i in range(self.max_length):
            position = torch.ones(encoder_output.shape[0], dtype=torch.int64) * i

            # Update mask
            self.mask = F.one_hot(position, num_classes=self.max_length)

            masked_score = self.adj_prob[:, i, :] - 100000000. * self.mask
            prob = torch.distributions.Bernoulli(logits=masked_score)

            sampled_arr = prob.sample()                                         # shape (batch_size, max_length)

            self.samples.append(sampled_arr)                                    # shape (max_length, batch_size, max_length)
            self.mask_scores.append(masked_score)                               # shape (max_length, batch_size, max_length)
            self.entropy.append(prob.entropy())                                 # shape (max_length, batch_size, max_length)

        return self.samples, self.mask_scores, self.entropy