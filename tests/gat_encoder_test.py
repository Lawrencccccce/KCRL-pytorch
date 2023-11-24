import torch
import torch.nn as nn
import torch.functional as F
from models.encoder import GATEncoder
from helpers.config import get_config

config, _ = get_config()
config.batch_size = 16
config.max_length = 8
config.input_dimension = 64

gat_encoder = GATEncoder(config, True)
input_ = torch.zeros((config.batch_size, config.max_length, config.input_dimension))
output = gat_encoder.encode(input_)