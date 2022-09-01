"""File to test all the initialisation functions
"""
import numpy as np
from utils.utils import (
    initialize_with_zeros,
    initialize_randomly
)

def test_init_with_zeros():
    """Test the initialize_with_zeros function
    """
    dimension = 100
    n_neurons = 1
    weights, biais = initialize_with_zeros(input_dim=dimension, n_units=n_neurons)
    assert weights.shape == (n_neurons, dimension)
    assert np.array_equal(weights, np.zeros([n_neurons, dimension]))
    assert biais.shape == (n_neurons, 1)
    assert np.array_equal(biais, np.zeros([n_neurons, 1]))

def test_init_random():
    """Test the initialize_randomly function
    """
    np.random.seed(2)
    dimension = 10
    n_neurons = 1
    weights, biais = initialize_randomly(input_dim=dimension, n_units=n_neurons)
    verif = np.array(
        [[-0.00416758, -0.00056267, -0.02136196, 0.01640271, -0.01793436,
        -0.00841747, 0.00502881, -0.01245288, -0.01057952, -0.00909008]]
    )
    assert weights.shape == (n_neurons, dimension)
    assert np.allclose(weights, verif)
    assert biais.shape == (n_neurons, 1)
    assert np.array_equal(biais, np.zeros([n_neurons, 1]))
