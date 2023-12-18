import numpy as np
import platform
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

from datetime import datetime
from pytz import timezone
from torch.utils.data import DataLoader
from dataloader.dataset import CustomDataset
from helpers.config import get_config
from models import Actor, Critic
from helpers.lambda_utils import BIC_lambdas
from helpers.dir_utils import create_dir
from helpers.log_helper import LogHelper
from helpers.analyze_utils import convert_graph_int_to_adj_mat, graph_prunned_by_coef, \
                                  count_accuracy, graph_prunned_by_coef_2nd
from rewards import get_Reward

import matplotlib
matplotlib.use('Agg')



def initialise_config_and_dataset():
    # Input parameters, these can be modified as per the dataset used

    dataset_path = "C:/Users/z5261241/Desktop/PhD/Code/KCRL-pytorch/datasets/"    # datasets path
    dataset_path = "D:\PhD\Code\KCRL-pytorch/datasets/"
    data_path = '{}/LUCAS.npy'.format(dataset_path)                               # specific dataset
    solu_path = '{}/LUCAS_sol.npy'.format(dataset_path)                           # specific solution

    num_random_sample = 5

    config, _ = get_config()
    
    mydataset = CustomDataset(data_path, num_random_sample, solution_path= solu_path)
    
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

    # initialise Logger
    output_dir = 'output/{}'.format(datetime.now(timezone('Australia/Sydney')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
    create_dir(output_dir)
    LogHelper.setup(log_path='{}/training.log'.format(output_dir),
                    level_str='INFO')
    _logger = logging.getLogger(__name__)
    _logger.info('Python version is {}'.format(platform.python_version()))
    _logger.info('Current commit of code: ___')

    

    # initialise dataset
    
    config, mydataset = initialise_config_and_dataset()

    config.save_model_path = '{}/model'.format(output_dir)
    # config.restore_model_path = '{}/model'.format(output_dir)
    config.summary_dir = '{}/summary'.format(output_dir)
    config.plot_dir = '{}/plot'.format(output_dir)
    config.graph_dir = '{}/graph'.format(output_dir)

    # Create directory
    create_dir(config.summary_dir)
    create_dir(config.summary_dir)
    create_dir(config.plot_dir)
    create_dir(config.graph_dir)

    

    # initialise dataloader
    shuffle = True
    num_workers = 4
    data_loader = DataLoader(dataset=mydataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=num_workers)


    # initialise actor and critic
    actor = Actor(config)
    critic = Critic(config)
    

    # set up lambda parameters
    if config.lambda_flag_default:
        
        sl, su, strue = BIC_lambdas(mydataset.get_data(), None, None, mydataset.get_true_graph().T, config.reg_type, config.score_type)
        
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
        _logger.info('Original sl: {}, su: {}, strue: {}'.format(sl, su, strue))
        _logger.info('Transfomed sl: {}, su: {}, lambda2: {}, true: {}'.format(sl, su, lambda2,
                     (strue-sl)/(su-sl)*lambda1_upper))
        
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
    global_step = torch.nn.Parameter(torch.Tensor([0]), requires_grad=False)  # global step
    global_step2 = torch.nn.Parameter(torch.Tensor([0]), requires_grad=False)  # global step

    prior_knowledge_g = get_prior_knowledge_graph(config.max_length)

    # Training loop
    for i in (range(1, config.nb_epoch + 1)):
        
        
        # batch = next(data_iter)
        batch = mydataset.get_random_batch(config.batch_size)
        batch = batch.float()
        encoder_output, graph_predict, log_softmax, entropy_regularization = actor.forward(batch)
        predicted_reward = critic.forward(encoder_output)
        # print(graph_predict[0])
        # break

        # reward_feed: (batch_size, [reward, score, cycness, penalty])
        reward_feed = torch.from_numpy(callreward.cal_rewards(graph_predict, prior_knowledge_g, lambda1, lambda2, lambda3))
        
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
        reward_batch_score_cyc = torch.mean(reward_feed[:,1:], axis=0)

        
        reward_mean = torch.mean(-reward_feed[:,0])
        reward_batch = reward_mean
        avg_baseline = config.alpha * config.init_baseline + (1.0 - config.alpha) * reward_mean
        
        
        # Actor update
        lr1 = config.lr1_start * (config.lr1_decay_rate ** (global_step / config.lr1_decay_step))
        opt1 = optim.Adam(list(actor.encoder.parameters()) + list(actor.decoder.parameters()), lr=lr1.item(), betas=(0.9, 0.99), eps=1e-7)
        reward_baseline = -reward_feed[:,0] - avg_baseline - predicted_reward
        # loss1 = torch.mean(reward_baseline * log_softmax) - 1 * lr1 * torch.mean(entropy_regularization)
        loss1 = -reward_mean
        print(loss1)
        # Critic update
        lr2 = config.lr2_start * (config.lr2_decay_rate ** (global_step2 / config.lr2_decay_step))
        opt2 = optim.Adam(critic.parameters(), lr=lr2.item(), betas=(0.9, 0.99), eps=1e-7)
        weights_ = 1.0
        # loss2 = torch.mean((-reward_feed[:,0] - avg_baseline - predicted_reward) ** 2)
        loss2 = torch.mean((-reward_feed[:,0] - predicted_reward) ** 2)
        # print(loss2)

        opt1.zero_grad()
        opt2.zero_grad()
        (loss1 + loss2).backward()
        torch.nn.utils.clip_grad_norm_(list(actor.encoder.parameters()) + list(actor.decoder.parameters()), 1)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1)
        opt1.step()
        opt2.step()
        global_step += 1
        global_step2 += 1



        lambda1s.append(lambda1)
        lambda2s.append(lambda2)
        lambda3s.append(lambda3)

        rewards_avg_baseline.append(avg_baseline)
        rewards_batches.append(reward_batch_score_cyc.numpy())
        reward_max_per_batch.append(max_reward_batch_score_cyc)

        graphss.append(torch.mean(graph_predict, axis=0))
        probsss.append(log_softmax)
        max_rewards.append(max_reward_score_cyc)



        # logging
        if i == 1 or i % 500 == 0:
            # if i >= 500:
            #     writer.add_summary(summary,i)
            print(f"Epoch {i}")
            _logger.info('[iter {}] reward_batch: {}, max_reward: {}, max_reward_batch: {}'.format(i,
                         reward_batch, max_reward, max_reward_batch))
            # other logger info; uncomment if you want to check
            # _logger.info('graph_batch_avg: {}'.format(graph_batch))
            _logger.info('graph true:\n {}'.format(mydataset.get_true_graph()))
            # _logger.info('graph weights true: {}'.format(training_set.b))
            _logger.info('=====================================')

            plt.figure(1)
            plt.plot(rewards_batches, label='reward per batch')
            plt.plot(max_rewards, label='max reward')
            plt.legend()
            plt.savefig('{}/reward_batch_average.png'.format(config.plot_dir))
            plt.close()

            image_count += 1
            # this draw the average graph per batch. 
            # can be modified to draw the graph (with or w/o pruning) that has the best reward
            fig = plt.figure(2)
            fig.suptitle('Iteration: {}'.format(i))
            ax = fig.add_subplot(1, 2, 1)
            ax.set_title('recovered_graph')
            ax.imshow(np.around(torch.mean(graph_predict, axis=0).T).numpy().astype(int),cmap=plt.cm.gray)
            ax = fig.add_subplot(1, 2, 2)
            ax.set_title('ground truth')
            ax.imshow(mydataset.get_true_graph(), cmap=plt.cm.gray)
            plt.savefig('{}/recovered_graph_iteration_{}.png'.format(config.plot_dir, image_count))
            plt.close()



        # update lambda1, lamda2, lamda3
        if (i+1) % lambda_iter_num == 0:
            ls_kv = callreward.update_all_scores(lambda1, lambda2, lambda3)
            # np.save('{}/solvd_dict_epoch_{}.npy'.format(config.graph_dir, i), np.array(ls_kv))
            max_rewards_re = callreward.update_scores(max_rewards, lambda1, lambda2, lambda3)
            rewards_batches_re = callreward.update_scores(rewards_batches, lambda1, lambda2, lambda3)
            reward_max_per_batch_re = callreward.update_scores(reward_max_per_batch, lambda1, lambda2, lambda3)

            # saved somewhat more detailed logging info
            # np.save('{}/solvd_dict.npy'.format(config.graph_dir), np.array(ls_kv))
            pd.DataFrame(np.array(max_rewards_re)).to_csv('{}/max_rewards.csv'.format(output_dir))
            pd.DataFrame(rewards_batches_re).to_csv('{}/rewards_batch.csv'.format(output_dir))
            pd.DataFrame(reward_max_per_batch_re).to_csv('{}/reward_max_batch.csv'.format(output_dir))
            pd.DataFrame(lambda1s).to_csv('{}/lambda1s.csv'.format(output_dir))
            pd.DataFrame(lambda2s).to_csv('{}/lambda2s.csv'.format(output_dir))
            pd.DataFrame(lambda3s).to_csv('{}/lambda3s.csv'.format(output_dir))
                
            graph_int, score_min, cyc_min = np.int32(ls_kv[0][0]), ls_kv[0][1][1], ls_kv[0][1][-1]

            if cyc_min < 1e-5:
                lambda1_upper = score_min
            lambda1 = min(lambda1+lambda1_update_add, lambda1_upper)
            lambda2 = min(lambda2*lambda2_update_mul, lambda2_upper)
            lambda3 = min(lambda3+lambda3_update_add, lambda3_upper)
            # _logger.info('[iter {}] lambda1 {}, upper {}, lambda2 {}, upper {}, score_min {}, cyc_min {}'.format(i+1,
            #             lambda1, lambda1_upper, lambda2, lambda2_upper, score_min, cyc_min))
                
            graph_batch = convert_graph_int_to_adj_mat(graph_int)

            if config.reg_type == 'LR':
                graph_batch_pruned = np.array(graph_prunned_by_coef(graph_batch, mydataset.get_data()))
            elif config.reg_type == 'QR':
                graph_batch_pruned = np.array(graph_prunned_by_coef_2nd(graph_batch, mydataset.get_data()))
            elif config.reg_type == 'GPR':
                # The R codes of CAM pruning operates the graph form that (i,j)=1 indicates i-th node-> j-th node
                # so we need to do a tranpose on the input graph and another tranpose on the output graph
                #graph_batch_pruned = np.array(graph_prunned_by_coef_2nd(graph_batch, training_set.inputdata))
                pass
                # graph_batch_pruned = np.transpose(pruning_cam(mydataset.get_data(), np.array(graph_batch).T))
            
            image_count2 += 1

            fig = plt.figure(3)
            fig.suptitle('Iteration: {}'.format(i))
            ax = fig.add_subplot(1, 2, 1)
            ax.set_title('est_graph')
            ax.imshow(np.around(graph_batch_pruned.T).astype(int),cmap=plt.cm.binary)
            ax = fig.add_subplot(1, 2, 2)
            ax.set_title('true_graph')
            ax.imshow(mydataset.get_true_graph(), cmap=plt.cm.binary)
            plt.savefig('{}/estimated_graph_{}.png'.format(config.plot_dir, image_count2))
            plt.close()

            # estimate accuracy
            acc_est = count_accuracy(mydataset.get_true_graph(), graph_batch.T)
            acc_est2 = count_accuracy(mydataset.get_true_graph(), graph_batch_pruned.T)

            fdr, tpr, fpr, shd, nnz = acc_est['fdr'], acc_est['tpr'], acc_est['fpr'], acc_est['shd'], \
                                        acc_est['pred_size']
            fdr2, tpr2, fpr2, shd2, nnz2 = acc_est2['fdr'], acc_est2['tpr'], acc_est2['fpr'], acc_est2['shd'], \
                                            acc_est2['pred_size']
                
            accuracy_res.append((fdr, tpr, fpr, shd, nnz))
            accuracy_res_pruned.append((fdr2, tpr2, fpr2, shd2, nnz2))
            
            np.save('{}/accuracy_res.npy'.format(output_dir), np.array(accuracy_res))
            np.save('{}/accuracy_res2.npy'.format(output_dir), np.array(accuracy_res_pruned))
                
            _logger.info('before pruning: fdr {}, tpr {}, fpr {}, shd {}, nnz {}'.format(fdr, tpr, fpr, shd, nnz))
            _logger.info('after  pruning: fdr {}, tpr {}, fpr {}, shd {}, nnz {}'.format(fdr2, tpr2, fpr2, shd2, nnz2))

            # Save the variables to disk
            # if i % max(1, int(config.nb_epoch / 5)) == 0 and i != 0:
                # curr_model_path = saver.save(sess, '{}/tmp.ckpt'.format(config.save_model_path), global_step=i)
                #_logger.info('Model saved in file: {}'.format(curr_model_path))

        #_logger.info('Training COMPLETED !')
        #saver.save(sess, '{}/actor.ckpt'.format(config.save_model_path))






if __name__ == "__main__":
    main()