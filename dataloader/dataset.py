
import numpy as np
from torch.utils.data import Dataset
from datetime import datetime
from pytz import timezone
from helpers.dir_utils import create_dir
import torch



class CustomDataset(Dataset):
    def __init__(self, input_path, num_random_sample = 64, transform=None, solution_path = None, normalize_flag=False, transpose_flag=False):
        self.data = np.load(input_path).astype(int)
        self.transform = transform
        self.datasize, self.d = self.data.shape
        self.num_random_sample = num_random_sample

        # data_dir = 'dataset/{}'.format(datetime.now(timezone('Australia/Sydney')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
        # create_dir(data_dir)


        if solution_path is None:
            true_graph = np.zeros((self.d, self.d))
            # self.true_graph = None
        else:
            true_graph = np.load(solution_path)#DAG.npy
            if transpose_flag: 
                true_graph = np.transpose(true_graph)#Transposing the true DAG
        self.true_graph = np.int32(np.abs(true_graph) > 1e-3)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # we random sample data from the dataset
        # return data in the shape of (number of nodes, number of samples)
        seq = np.random.randint(self.datasize, size=(self.num_random_sample))
        input_ = self.data[seq]
        
        return input_.T
    
    def get_random_batch(self, batch_size):
        output = []
        for _ in range(batch_size):
            seq = np.random.randint(self.datasize, size=(self.num_random_sample))
            input_ = self.data[seq]
            output.append(input_.T)
        
        return torch.tensor(output)
    
    def get_number_of_nodes(self):
        return self.d
    
    def get_datasize(self):
        return self.datasize
    
    def get_data(self):
        return self.data
    
    def get_true_graph(self):
        return self.true_graph