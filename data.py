import torch
import numpy as np
import yaml


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, config):
        self.data = data
        self.frames = config["frames"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # sourcery skip: inline-immediately-returned-variable
        gif = self.data[index]

        gif_split = {"old_frames": gif[:self.frames],
                     "new_frames": gif[self.frames:]}

        return gif_split


def prepare_data(config):
    """Data preparation function.

    Args:
        config (yaml): Typical config file.

    Returns:
        train_dataset, valid_dataset, test_dataset (torch.utils.data.Dataset): Datasets for training, validation and testing.

    """
    np.random.seed(config["seed"])

    data = np.load(config["path"])
    data = np.transpose(data, (1, 0, 2, 3))

    np.random.shuffle(data)

    train_dataset = data[:8000]
    valid_dataset = data[8000:9000]
    test_dataset = data[9000:]

    return train_dataset, valid_dataset, test_dataset


def get_data_loader(config):
    """Data loader function.
    Args:   
        config (yaml): Typical config file.

    Returns:
        train_loader, valid_loader, test_loader (torch.utils.data.DataLoader): Dataloaders for training, validation and testing.

    """

    train, valid, test = prepare_data(config)
    train_dataset = torch.from_numpy(train).float()
    valid_dataset = torch.from_numpy(valid).float()
    test_dataset = torch.from_numpy(test).float()

    train_dataset = Dataset(train_dataset, config)
    valid_dataset = Dataset(valid_dataset, config)
    test_dataset = Dataset(test_dataset, config)

    params = {'batch_size': config["batch_size"],
              'shuffle': config["shuffle"], 'num_workers': config["num_workers"]}

    train_loader = torch.utils.data.DataLoader(train_dataset, **params)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, **params)
    test_loader = torch.utils.data.DataLoader(test_dataset, **params)

    return train_loader, valid_loader, test_loader
