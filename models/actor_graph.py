import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from .encoder import GATEncoder
from .decoder import SingleLayerDecoder
from critic import Critic



def variable_summaries(name, var, with_max_min=False):
    mean = torch.mean(var)
    print(f'{name} mean: {mean.item()}')

    stddev = torch.sqrt(torch.mean((var - mean) ** 2))
    print(f'{name} stddev: {stddev.item()}')

    if with_max_min:
        max_val = torch.max(var)
        min_val = torch.min(var)
        print(f'{name} max: {max_val.item()}')
        print(f'{name} min: {min_val.item()}')




class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        self.config = config
        self.is_train = True
        # Data config
        self.batch_size = config.batch_size  # batch size
        self.max_length = config.max_length  
        self.input_dimension = config.input_dimension  

        # Reward config
        self.avg_baseline = nn.Parameter(torch.Tensor([config.init_baseline]), requires_grad=False)
        self.alpha = config.alpha  # moving average update

        # Training config (actor)
        self.global_step = nn.Parameter(torch.Tensor([0]), requires_grad=False)  # global step
        self.lr1_start = config.lr1_start  # initial learning rate
        self.lr1_decay_rate = config.lr1_decay_rate  # learning rate decay rate
        self.lr1_decay_step = config.lr1_decay_step  # learning rate decay step

        # Training config (critic)
        self.global_step2 = nn.Parameter(torch.Tensor([0]), requires_grad=False)  # global step
        self.lr2_start = config.lr1_start  # initial learning rate
        self.lr2_decay_rate = config.lr1_decay_rate  # learning rate decay rate
        self.lr2_decay_step = config.lr1_decay_step  # learning rate decay step

        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        self.input_ = torch.zeros((self.batch_size, self.max_length, self.input_dimension))
        self.reward_ = torch.zeros(self.batch_size)
        self.graphs_ = torch.zeros((self.batch_size, self.max_length, self.max_length))



        self.build_permutation()
        self.build_critic()
        self.build_reward()
        self.build_optim()
        #self.merged = tf.summary.merge_all()


    def build_permutation(self):
        # if self.config.encoder_type == 'TransformerEncoder':
        #     encoder = TransformerEncoder(self.config, self.is_train)
        # elif self.config.encoder_type == 'GATEncoder':
        #     encoder = GATEncoder(self.config, self.is_train)
        # else:
        #     raise NotImplementedError('Current encoder type is not implemented yet!')

        # for now we just use GATEncoder as encoder
        self.encoder = GATEncoder(self.config, self.is_train)
        self.encoder_output = self.encoder.encode(self.input_)

        # if self.config.decoder_type == 'SingleLayerDecoder':
        #     self.decoder = SingleLayerDecoder(self.config, self.is_train)
        # elif self.config.decoder_type == 'TransformerDecoder':
        #     self.decoder = TransformerDecoder(self.config, self.is_train)
        # elif self.config.decoder_type == 'BilinearDecoder':
        #     self.decoder = BilinearDecoder(self.config, self.is_train)
        # elif self.config.decoder_type == 'NTNDecoder':
        #     self.decoder = NTNDecoder(self.config, self.is_train)
        # else:
        #     raise NotImplementedError('Current decoder type is not implemented yet!')


        # for now we just use SingleLayerDecoder as decoder
        self.decoder = SingleLayerDecoder(self.config, self.is_train)
        self.samples, self.scores, self.entropy = self.decoder.decode(self.encoder_output)


        graphs_gen = torch.stack(self.samples).permute(1, 0, 2)                         # shape (batch_size, max_length, max_length)

        self.graphs = graphs_gen
        # average the graphs_gen over all batches
        self.graph_batch = torch.mean(graphs_gen, axis=0)                               # shape (max_length, max_length)
        logits_for_rewards = torch.stack(self.scores).permute(1, 0, 2)                  # shape (batch_size, max_length, max_length)
        entropy_for_rewards = torch.stack(self.entropy).permute(1, 0, 2)                # shape (batch_size, max_length, max_length)    
        # why only the first two?
        self.test_scores = torch.sigmoid(logits_for_rewards)[:2]
        log_probss = F.binary_cross_entropy_with_logits(logits_for_rewards, self.graphs_, reduction='none')
        self.log_softmax = torch.mean(log_probss, axis=[1, 2])
        self.entropy_regularization = torch.mean(entropy_for_rewards, axis=[1,2])


    def build_critic(self):
        # Critic predicts reward (parametric baseline for REINFORCE)
        self.critic = Critic(self.config, self.is_train)
        self.critic.predict_rewards(self.encoder_output)

    def build_reward(self):
        # PyTorch doesn't have a direct equivalent of tf.name_scope
        # But you can achieve similar functionality using Python's context managers
        self.reward = self.reward_
        # PyTorch doesn't have a direct equivalent of tf.summary
        # You can use TensorBoardX, a third-party library, to log values for viewing in TensorBoard
        # from tensorboardX import SummaryWriter
        # writer = SummaryWriter()
        # writer.add_scalar('reward', self.reward)

    def build_optim(self):
        # PyTorch doesn't have a direct equivalent of tf.control_dependencies
        # But you can achieve similar functionality using Python's context managers

        # Update baseline
        reward_mean = torch.mean(self.reward)
        self.reward_batch = reward_mean
        self.avg_baseline = self.alpha * self.avg_baseline + (1.0 - self.alpha) * reward_mean

        # Actor learning rate
        self.lr1 = self.lr1_start * (self.lr1_decay_rate ** (self.global_step / self.lr1_decay_step))
        # Optimizer
        self.opt1 = optim.Adam(self.parameters(), lr=self.lr1, betas=(0.9, 0.99), eps=1e-7)
        # Discounted reward
        self.reward_baseline = self.reward - self.avg_baseline - self.critic.predictions
        # Loss
        self.loss1 = torch.mean(self.reward_baseline * self.log_softmax) - 1 * self.lr1 * torch.mean(self.entropy_regularization)
        # Minimize step
        self.opt1.zero_grad()
        self.loss1.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.opt1.step()
        self.global_step += 1

        # Critic learning rate
        self.lr2 = self.lr2_start * (self.lr2_decay_rate ** (self.global_step2 / self.lr2_decay_step))
        # Optimizer
        self.opt2 = optim.Adam(self.parameters(), lr=self.lr2, betas=(0.9, 0.99), eps=1e-7)
        # Loss
        weights_ = 1.0
        self.loss2 = torch.mean((self.reward - self.avg_baseline - self.critic.predictions) ** 2)
        # Minimize step
        self.opt2.zero_grad()
        self.loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.opt2.step()
        self.global_step2 += 1