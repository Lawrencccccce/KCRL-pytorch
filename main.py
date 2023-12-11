import numpy as np

from torch.utils.data import DataLoader
from dataloader.dataset import CustomDataset
from helpers.config import get_config
from models import Actor, Critic
from helpers.lambda_utils import BIC_lambdas
from rewards import get_Reward


def initialise_config_and_dataset():
    # Input parameters, these can be modified as per the dataset used

    dataset_path = "C:/Users/z5261241/Desktop/PhD/Code/KCRL-pytorch/datasets/"    # datasets path
    dataset_path = "D:\PhD\Code\KCRL-pytorch/datasets/"
    data_path = '{}/LUCAS.npy'.format(dataset_path)                               # specific dataset

    num_random_sample = 5

    config, _ = get_config()
    
    mydataset = CustomDataset(data_path, num_random_sample)
    
    config.data_path = data_path
    config.max_length = mydataset.get_number_of_nodes()    # This is the total number of nodes in the graph
    config.data_size = mydataset.get_datasize()   # Sample size
    config.score_type = 'BIC'
    config.reg_type = 'LR'
    config.read_data = True
    config.transpose = False
    config.lambda_flag_default = True
    config.nb_epoch = 10000                 
    config.input_dimension = 64
    config.lambda_iter_num = 1000

    config.batch_size = 10
    config.num_random_sample = num_random_sample
    config.num_stacks = 2
    config.num_heads = 5
    config.hidden_dim = 20

    


    return config, mydataset

def get_prior_knowledge_graph(num_nodes):
    # Incorporation of existing information.
    # Prior knowledge set formation for a graph with 8 nodes (8 x 8 adjacency matrix). 
    # It can be changed for a graph with any number of nodes.
    # Here prior knowledge: there is a directed edge from node 2-->6 and from 3-->4. No edge exists between nodes 5 to 1 and 7 to 6.
    # In this code, the generated graph adjacency matrix by the encoder-decoder is transposed. 
    # Hence for accurate comparison, the prior knowledge set is also transposed.
    # Any number of prior edges can be used as per the experimental requirement.     

    true_g = np.zeros((num_nodes, num_nodes))*2
    a= np.int32(true_g)
    return a


def main():

    # initialise config and dataset
    

    # initialise dataset
    
    config, mydataset = initialise_config_and_dataset()

    # initialise dataloader
    shuffle = True
    num_workers = 4
    data_loader = DataLoader(dataset=mydataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=num_workers)


    # initialise actor and critic
    actor = Actor(config)
    critic = Critic(config)
    

    # set up lambda parameters
    if config.lambda_flag_default:
        
        sl, su, strue = BIC_lambdas(mydataset.get_data(), None, None, mydataset.get_true_graph().T if mydataset.get_true_graph() else None, config.reg_type, config.score_type)
        
        lambda1 = 0
        lambda1_upper = 5
        lambda1_update_add = 1
        lambda2 = 1/(10**(np.round(config.max_length/3)))
        lambda2_upper = 0.01
        lambda2_update_mul = 10
        lambda3 = 0
        lambda3_upper = 1
        lambda3_update_add = 0.1
        lambda_iter_num = config.lambda_iter_num

        # test initialized score
        # _logger.info('Original sl: {}, su: {}, strue: {}'.format(sl, su, strue))
        # _logger.info('Transfomed sl: {}, su: {}, lambda2: {}, true: {}'.format(sl, su, lambda2,
        #              (strue-sl)/(su-sl)*lambda1_upper))
        
    else:
        # test choices for the case with manually provided bounds
        # not fully tested

        sl = config.score_lower
        su = config.score_upper
        if config.score_bd_tight:
            lambda1 = 2
            lambda1_upper = 2
        else:
            lambda1 = 0
            lambda1_upper = 5
            lambda1_update_add = 1
        lambda2 = 1/(10**(np.round(config.max_length/3)))
        lambda2_upper = 0.01
        lambda2_update_mul = config.lambda2_update
        lambda3 = 0
        lambda3_upper = 1
        lambda3_update_add = 0.1
        lambda_iter_num = config.lambda_iter_num

    # get reward function
    callreward = get_Reward(actor.batch_size, config.max_length, actor.num_random_sample, mydataset.get_data(),
                            sl, su, lambda1_upper, config.score_type, config.reg_type, config.l1_graph_reg, False)    

    # Initialize useful variables
    rewards_avg_baseline = []
    rewards_batches = []
    reward_max_per_batch = []
    
    lambda1s = []
    lambda2s = []
    lambda3s = []
    
    graphss = []
    probsss = []
    max_rewards = []
    max_reward = float('-inf')
    image_count = 0
    image_count2= 0
    
    accuracy_res = []
    accuracy_res_pruned = []
    
    max_reward_score_cyc = (lambda1_upper+1, 0, 0)

    prior_knowledge_g = get_prior_knowledge_graph(config.max_length)

    # Training loop
    for i in (range(1, config.nb_epoch + 1)):
        for batch in data_loader:
            # print(batch)
            batch = batch.float()
            encoder_output, graph_predict, mask_scores, entropy = actor.forward(batch)
            predicted_reward = critic.forward(encoder_output)

            reward_feed = callreward.cal_rewards(graph_predict, prior_knowledge_g, lambda1, lambda2, lambda3)

            # max reward, max reward per batch
            max_reward = -callreward.update_scores([max_reward_score_cyc], lambda1, lambda2, lambda3)[0]
            max_reward_batch = float('inf')
            max_reward_batch_score_cyc = (0, 0, 0)

            for reward_, score_, cyc_, penalty_ in reward_feed:
                if reward_ < max_reward_batch:
                    max_reward_batch = reward_
                    max_reward_batch_score_cyc = (score_, cyc_, penalty_)
                        
            max_reward_batch = -max_reward_batch

            if max_reward < max_reward_batch:
                max_reward = max_reward_batch
                max_reward_score_cyc = max_reward_batch_score_cyc

            # for average reward per batch
            reward_batch_score_cyc = np.mean(reward_feed[:,1:], axis=0)

            print(graph_predict.shape)
            print(predicted_reward)
            print(reward_batch_score_cyc)
            break
        break







if __name__ == "__main__":
    main()