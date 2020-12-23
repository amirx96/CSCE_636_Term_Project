import os
import pickle
import numpy as np
import torch, torchvision
import torchvision.transforms as transforms
import utils
import PrivateDataset
"""This script implements the functions for reading data.
"""

def load_data(data_dir,train_aug):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        trainset: An torch tensor of [x_train, y_train] -> (50000,3,32,32), (50000,) 
            (dtype=np.float32)
        testset: An torch tensor of [x_test, y_test] -> (10000,3,32,32), (10000,) 
            (dtype=np.float32)
    """

    default_transform = transforms.Compose([ # defualt transform which only normalizes the data set
            transforms.ToTensor(),
            transforms.Normalize(utils.cifar_10_norm_mean, utils.cifar_10_norm_std) # normalize dataset
        ])

    default_train_transform = transforms.Compose([ # defualt transform which only normalizes the data set
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(utils.cifar_10_norm_mean, utils.cifar_10_norm_std) # normalize dataset
        ])   
    # transform with training augmentation
    if train_aug is not None:
        # train_transform = transforms.Compose([
        #     train_aug(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(utils.cifar_10_norm_mean, utils.cifar_10_norm_std) # normalize dataset
        #     ]
        # )
        train_transform = train_aug
    else:
        train_transform = default_train_transform

    trainset = torchvision.datasets.CIFAR10(
    root=data_dir, train=True, download=True, transform=train_transform)
    
    orig_trainset = torchvision.datasets.CIFAR10(
    root=data_dir, train=True, download=False, transform=default_transform) #original trainingset without augmentation

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=default_transform)

    return trainset,testset,orig_trainset


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 32, 32, 3].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    default_transform = transforms.Compose([ # defualt transform which only normalizes the data set
            transforms.ToTensor(),
            transforms.Normalize(utils.cifar_10_norm_mean, utils.cifar_10_norm_std) # normalize dataset
        ])
    x_test = PrivateDataset.CSCE_636_PrivateDataset(data_dir,transform=default_transform)
    ### END CODE HERE

    return x_test


def train_valid_split(train, orig_trainset, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        trainset: An torch tensor of [x_train, y_train] -> (50000,3,32,32), (50000,) 
            (dtype=np.float32)
        train_ratio: A float number between 0 and 1.

    Returns:
        TODO
    """
    if train_ratio == 1:
        return train,None

    train_size = int(train_ratio*len(train))
    valid_size = len(train) - train_size
    train,_ = torch.utils.data.random_split(train,[train_size,valid_size])
    _, valid = torch.utils.data.random_split(orig_trainset,[train_size,valid_size])
    return train,valid

