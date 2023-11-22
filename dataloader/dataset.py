import torch
import numpy as np
from torch.utils.data import Dataset
from datetime import datetime
from pytz import timezone
from helpers.dir_utils import create_dir





class CustomDataset(Dataset):
    def __init__(self, input_path, transform=None, solution_path = None, normalize_flag=False, transpose_flag=False):
        self.data = np.load(input_path)
        self.transform = transform

        data_dir = 'dataset/{}'.format(datetime.now(timezone('Australia/Sydney')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
        create_dir(data_dir)


        if solution_path is None:
            true_graph = np.zeros(self.d)
        else:
            true_graph = np.load(solution_path)#DAG.npy
            if transpose_flag: 
                true_graph = np.transpose(true_graph)#Transposing the true DAG
        self.true_graph = np.int32(np.abs(true_graph) > 1e-3)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample