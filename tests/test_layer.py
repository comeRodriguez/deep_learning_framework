"""This file is for testing the layer classes
"""
import numpy as np
from neural_networks.layer import FullyConnected
from utils.utils import load_planar_dataset

FEATURES, LABELS = load_planar_dataset(n_example=10)

def test_fully_connected_init():
    """Test the fully connected class initialization
    """
    dense_layer = FullyConnected(FEATURES.T, 2, "sigmoid", random_seed=2)
    expected_weights = np.array(
        [
            [-0.00416758, -0.00056267],
            [-0.02136196,  0.01640271]
        ]
    )
    assert dense_layer.weights.shape == (FEATURES.T.shape[0], 2)
    assert dense_layer.biais.shape == (2, 1)
    assert np.allclose(dense_layer.weights, expected_weights)
    assert np.array_equal(dense_layer.biais, np.zeros([2, 1]))

def test_fully_connected_linear_part():
    """Test the FullyConnected.compute_linear_part() method
    """
    dense_layer = FullyConnected(FEATURES.T, 2, "sigmoid", random_seed=2)
    dense_layer.compute_linear_part()
    expected_linear_part_values = np.array(
        [
            [-0.00632341, -0.00689881, 0.00819298, -0.01056383, 0.0026409,
                0.00549181, 0.00757634, -0.00708969, 0.00518389, -0.00367483
            ],
            [0.02961156, -0.00017528 , 0.03762948, -0.08719568, -0.02894833,
                -0.03343862, -0.03186331, -0.03319292, 0.04789327, 0.03419797
            ]
        ]
    )
    assert dense_layer.linear_part.shape == (2, 10)
    assert np.allclose(dense_layer.linear_part, expected_linear_part_values)

def test_fully_connected_activations_part():
    """Test the FullyConnected.compute_activations() method
    """
    dense_layer = FullyConnected(FEATURES.T, 2, "sigmoid", random_seed=2)
    dense_layer.compute_activations()
    expected_activation_values = np.array(
        [
            [0.49841915, 0.4982753, 0.50204823, 0.49735907, 0.50066022,
            0.50137295, 0.50189408, 0.49822758, 0.50129597, 0.49908129
            ],
            [0.50740235, 0.49995618, 0.50940626, 0.47821488, 0.49276342,
            0.49164112, 0.49203485, 0.49170253, 0.51197103, 0.50854866
            ]
        ]
    )
    assert dense_layer.activations.shape == (2, 10)
    assert np.allclose(dense_layer.activations, expected_activation_values)
