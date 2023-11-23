import torch
import torch.nn as nn
import torch.nn.functional as F

def multihead_attention(inputs, num_units=None, num_heads=16, dropout_rate=0.1, is_training=True):
    Q = nn.Linear(inputs.size(-1), num_units)(inputs)
    K = nn.Linear(inputs.size(-1), num_units)(inputs)
    V = nn.Linear(inputs.size(-1), num_units)(inputs)

    Q_ = torch.cat(Q.chunk(num_heads, dim=2), dim=0)
    K_ = torch.cat(K.chunk(num_heads, dim=2), dim=0)
    V_ = torch.cat(V.chunk(num_heads, dim=2), dim=0)

    outputs = torch.bmm(Q_, K_.transpose(1, 2))

    outputs = outputs / (K_.size(-1) ** 0.5)

    outputs = F.softmax(outputs, dim=-1)

    return outputs









def feedforward(inputs, num_units=[2048, 512], is_training=True):
    # Inner layer
    outputs = nn.Conv1d(inputs.size(1), num_units[0], kernel_size=1)(inputs)
    outputs = F.relu(outputs)

    # Readout layer
    outputs = nn.Conv1d(outputs.size(1), num_units[1], kernel_size=1)(outputs)

    # Residual connection
    outputs += inputs

    # Normalize
    outputs = nn.BatchNorm1d(outputs.size(1))(outputs)

    return outputs



class TransformerEncoder(nn.Module):
    def __init__(self, config, is_train):
        super(TransformerEncoder, self).__init__()
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.input_dimension = config.input_dimension
        self.input_embed = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_stacks = config.num_stacks
        self.is_training = is_train
        self.W_embed = nn.Parameter(torch.Tensor(1, self.input_dimension, self.input_embed))
        nn.init.xavier_uniform_(self.W_embed)

    def encode(self, inputs):
        self.embedded_input = F.conv1d(inputs, self.W_embed, stride=1)
        self.enc = nn.BatchNorm1d(self.embedded_input.size(1))(self.embedded_input)

        for i in range(self.num_stacks):
            self.enc = multihead_attention(self.enc, num_units=self.input_embed, num_heads=self.num_heads, dropout_rate=0.0, is_training=self.is_training)
            self.enc = feedforward(self.enc, num_units=[4*self.input_embed, self.input_embed], is_training=self.is_training)

        return self.enc