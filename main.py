from torch.utils.data import DataLoader
from dataloader.dataset import CustomDataset
from helpers.config import get_config


def initialise_config_and_dataset():
    # Input parameters, these can be modified as per the dataset used

    dataset_path = "C:/Users/z5261241/Desktop/PhD/Code/KCRL-pytorch/datasets/"    # datasets path
    data_path = '{}/LUCAS.npy'.format(dataset_path)                               # specific dataset

    mydataset = CustomDataset(data_path)

    config, _ = get_config()

    
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

    return config, mydataset


def main():

    # initialise config and dataset
    

    # initialise dataset
    
    config, mydataset = initialise_config_and_dataset()


    batch_size = 16
    shuffle = True
    num_workers = 4

    data_loader = DataLoader(dataset=mydataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    for batch in data_loader:
        input_data = batch['data']
        print(input_data)







if __name__ == "__main__":
    main()