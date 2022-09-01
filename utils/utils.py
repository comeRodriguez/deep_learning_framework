"""This file contains usefull functions as initialization for example
"""
from typing import Tuple
import numpy as np

def initialize_with_zeros(input_dim: int, n_units: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function creates a vector of zeros of shape (n_units, input_dim) for w and
    initializes b to a zero vector of shape (n_units, 1).

    Args:
        input_dim (int): dimension of the feature vector
        n_units (int): number of neurons in the layer

    Returns:
        Tuple[np.ndarray, int]: W matrix and b vector
    """
    weights = np.zeros((n_units, input_dim))
    biais = np.zeros([n_units, 1])
    return weights, biais

def initialize_randomly(input_dim: int, n_units: int, random_seed: int=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function creates a random vector of shape (n_units, input_dim) for w and
    initializes b to a zero vector of shape (n_units, 1).

    Args:
        input_dim (int): dimension of the feature vector
        n_units (int): number of neurons in the layer
        random_seed (int): if presents, let the code generate always same values

    Returns:
        Tuple[np.ndarray, int]: W matrix and b vector
    """
    if random_seed:
        np.random.seed(random_seed)
    weights = np.random.randn(n_units, input_dim) * 0.01
    biais = np.zeros([n_units, 1])
    return weights, biais

def load_planar_dataset(n_example: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create and return a mock dataset. If we plot the data, we can see
    a sort of flower. The created dataset contains n_example X of 2 features
    and their associated label

    Args:
        n_example (int): number of examples in the wanted dataset

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features matrix and associated labels vector
    """
    np.random.seed(1)
    N = int(n_example/2)
    D = 2
    FEATURES = np.zeros((n_example,D))
    LABELS = np.zeros((n_example,1), dtype='uint8')
    max_ray = 4
    for label in range(2):
        index = range(N*label,N*(label+1))
        theta = np.linspace(label*3.12,(label+1)*3.12,N) + np.random.randn(N)*0.2
        radius = max_ray*np.sin(4*theta) + np.random.randn(N)*0.2
        FEATURES[index] = np.c_[radius*np.sin(theta), radius*np.cos(theta)]
        LABELS[index] = label
    return FEATURES, LABELS
