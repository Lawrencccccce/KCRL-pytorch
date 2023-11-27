import torch
import torch.nn as nn
import torch.functional as F
from models.decoder import SingleLayerDecoder
from models.encoder import GATEncoder
from helpers.config import get_config


config, _ = get_config()
config.batch_size = 10
config.max_length = 12
config.input_dimension = 32
config.num_stacks = 2
config.num_heads = 5
config.hidden_dim = 20

gat_encoder = GATEncoder(config, True)
input_ = torch.zeros((config.batch_size, config.max_length, config.input_dimension))
output = gat_encoder.encode(input_)

decoder = SingleLayerDecoder(config, True)
samples, scores, entropy = decoder.decode(output)
print(samples)