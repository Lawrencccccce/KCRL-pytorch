from torch.utils.data import DataLoader
from dataloader.dataset import CustomDataset


def main():
    dataset_path = "C:/Users/z5261241/Desktop/PhD/Code/KCRL-pytorch/datasets/"
    data_path = '{}/LUCAS.npy'.format(dataset_path)

    mydataset = CustomDataset(data_path)


    batch_size = 16
    shuffle = True
    num_workers = 4

    data_loader = DataLoader(dataset=mydataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    for batch in data_loader:
        input_data = batch['data']
        print(input_data)







if __name__ == "__main__":
    main()