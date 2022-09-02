"""File to test all the cost functions and their derivatives
"""
import re
import numpy as np
import pytest
from keras.losses import BinaryCrossentropy
from neural_networks.cost import (
    binary_cross_entropy,
    binary_cross_entropy_derivative,
    CostFunction
)

def test_binary_cross_entropy():
    """Test the binary_cross_entropy cost function
    """
    mock_labels = np.array(
        [
            [1],
            [1],
            [0],
            [0]
        ]
    )
    mock_predicted_labels = np.array(
        [
            [0.36],
            [0.84],
            [0.52],
            [0.18]
        ]
    )
    cost = binary_cross_entropy(true_labels=mock_labels, predicted_labels=mock_predicted_labels)
    real_cost = BinaryCrossentropy()
    real_cost_value = real_cost(mock_labels, mock_predicted_labels).numpy()
    assert np.allclose(cost, real_cost_value)

def test_binary_cross_entropy_derivative():
    """Test the derivative of the binary_cross_entropy cost function
    """
    mock_labels = np.array(
        [
            [1],
            [1],
            [0],
            [0]
        ]
    )
    mock_predicted_labels = np.array(
        [
            [0.36],
            [0.84],
            [0.52],
            [0.18]
        ]
    )
    d_cost = binary_cross_entropy_derivative(
        true_labels=mock_labels,
        predicted_labels=mock_predicted_labels
    )
    assert np.allclose(d_cost, -0.1663521099)

def test_creation_of_cost_function():
    """Test the creation of cost function and the raise ValueError
    """
    mock_labels = np.array(
        [
            [1],
            [1],
            [0],
            [0]
        ]
    )
    mock_predicted_labels = np.array(
        [
            [0.36],
            [0.84],
            [0.52],
            [0.18]
        ]
    )
    cost = CostFunction(name="binary_cross_entropy")
    cost_authorized = ["binary_cross_entropy"]
    val = cost.get_computed_cost(true_labels=mock_labels, predicted_labels=mock_predicted_labels)
    d_val = cost.get_derivative_computed_cost(mock_labels, mock_predicted_labels)
    assert cost.get_name() == "binary_cross_entropy"
    with pytest.raises(
        ValueError,
        match=re.escape(f"Invalid cost function, expected one of {cost_authorized}")
    ):
        CostFunction(name="it's me")
    real_cost = BinaryCrossentropy()
    real_cost_value = real_cost(mock_labels, mock_predicted_labels).numpy()
    assert np.allclose(val, real_cost_value)
    assert np.allclose(d_val, -0.1663521099)
