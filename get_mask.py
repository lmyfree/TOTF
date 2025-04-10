import numpy as np
from numpy.random import randint
import torch
from sklearn.preprocessing import OneHotEncoder

def z_score_normalize(data):
    print(data.shape)
    #data = data.matrix()
    mean = torch.mean(data, axis=0)
    std_dev = torch.std(data, axis=0)
    normalized_data = (data - mean) / std_dev
    return normalized_data
def mean1(data):
    print(data.shape[0])
    #data = data.matrix()
    mean = torch.mean(data, axis=0)
    mean = mean.unsqueeze(1)
    print(mean.shape)
    mean1 = np.repeat(mean, data.shape[0], axis=1)
    print(mean1.shape)
    
    #std_dev = torch.std(data, axis=0)
    #normalized_data = (data - mean) / std_dev
    return mean1
def generate_uniform_mask(features, missing_rate):
    """

    Parameters
    ----------
    features : torch.tensor
    missing_rate : float

    Returns
    -------
    mask : torch.tensor
        mask[i][j] is True if features[i][j] is missing.

    """
    t = features.shape
    mask = torch.rand(size=features.shape)
    print(mask)
    mask = mask <= missing_rate
    return mask
def generate_uniform_mask1(features, missing_rate):
    """

    Parameters
    ----------
    features : torch.tensor
    missing_rate : float

    Returns
    -------
    mask : torch.tensor
        mask[i][j] is True if features[i][j] is missing.

    """
    mask = torch.rand(size=features.size())
    print(mask)
    mask = mask <= missing_rate
    return mask
def get_mask(view_num, data_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data.

        Args:
          view_num: view number
          data_len: number of samples
          missing_rate: Defined in section 4.1 of the paper
        Returns:
          mask

    """
    missing_rate = missing_rate / view_num
    one_rate = 1.0 - missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(data_len, view_num))
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
        one_num = view_num * data_len * one_rate - data_len
        ratio = one_num / (view_num * data_len)
        matrix_iter = (randint(0, 100, size=(data_len, view_num)) < int(ratio * 100)).astype(np.int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * data_len)
        matrix_iter = (randint(0, 100, size=(data_len, view_num)) < int(ratio * 100)).astype(np.int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
        ratio = np.sum(matrix) / (view_num * data_len)
        error = abs(one_rate - ratio)

    return matrix
