from torch.utils.data import DataLoader
from dataloader.dataset import CustomDataset
from helpers.config import get_config
from models import Actor


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


def main():

    # initialise config and dataset
    

    # initialise dataset
    
    config, mydataset = initialise_config_and_dataset()


    shuffle = True
    num_workers = 4

    actor = Actor(config)
    data_loader = DataLoader(dataset=mydataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=num_workers)
    for batch in data_loader:
        # print(batch)
        batch = batch.float()
        output = actor.forward(batch)

        print(output[0].shape)
        break







if __name__ == "__main__":
    main()