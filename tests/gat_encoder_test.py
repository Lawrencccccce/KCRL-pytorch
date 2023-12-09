import torch
import torch.nn as nn
import torch.functional as F
from models.encoder import GATEncoder
from models import Critic
from helpers.config import get_config

config, _ = get_config()
config.batch_size = 10
config.max_length = 5
config.num_random_sample = 32
config.num_stacks = 2
config.num_heads = 5
config.hidden_dim = 20

gat_encoder = GATEncoder(config, True)
critic = Critic(config)
input_ = torch.zeros((config.batch_size, config.max_length, config.input_dimension))
output = gat_encoder.forward(input_)
print(output.shape)

reward = critic.forward(output)
print(reward)