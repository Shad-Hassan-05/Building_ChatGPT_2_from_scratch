import torch
from torch.utils.data import TensorDataset


def load_dataset(dataset_path, mean_subtraction, normalization):
    """
    Reads the train and validation data

    Arguments
    ---------
    dataset_path: (string) representing the file path of the dataset
    mean_subtraction: (boolean) specifies whether to do mean centering or not. Default: False
    normalization: (boolean) specifies whether to normalizes the data or not. Default: False

    Returns
    -------
    train_ds (TensorDataset): The features and their corresponding labels bundled as a dataset
    """
    # Load the dataset and extract the features and the labels
    data = torch.load(dataset_path)
    features = data["features"] # Txn tensor
    labels = data["labels"] # 1xn vector

    # Do mean_subtraction if it is enabled
    if mean_subtraction:
        mean = features.mean(dim=0, keepdim=True) # 1xn vector ith element is the mean of the ith column of X 
        features = features - mean
        
    # do normalization if it is enabled
    if normalization:
        std = features.std(dim=0, keepdim=True) # 1xn vector of standard deviations
        mask = std

        # any feature with 0 std will have no effect by changing to 1.0 std since it will result in division by 1.0
        std[mask == 0] = 1
        features = (features / std)
            
    # create tensor dataset train_ds
    train_ds = TensorDataset(features, labels) # loads tensors into a data set

    return train_ds
