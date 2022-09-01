"""File to test all the activation functions and their derivatives
"""
import re
import numpy as np
import pytest
from neural_networks.activation_functions import (
    sigmoid,
    sigmoid_derivative,
    relu,
    relu_derivative,
    ActivationFunction
)

def test_sigmoid():
    """Test the sigmoid function
    """
    sig_value = sigmoid(0)
    sig_value_2 = sigmoid(np.array([0, 2]))
    sig_value_3 = sigmoid(np.array([[0, 2], [1, 3]]))
    assert sig_value == 0.5
    assert np.allclose(sig_value_2, np.array([0.5, 0.88079708]))
    assert np.allclose(sig_value_3, np.array([[0.5, 0.88079708], [0.73105858, 0.95257413]]))

def test_derivative_sigmoid():
    """Test the derivative of the sigmoid function
    """
    ds_value = sigmoid_derivative(0)
    ds_value_2 = sigmoid_derivative(np.array([0, 2]))
    ds_value_3 = sigmoid_derivative(np.array([[0, 2], [1, 3]]))
    assert ds_value == 0.25
    assert np.allclose(ds_value_2, np.array([0.25, 0.10499356]))
    assert np.allclose(ds_value_3, np.array([[0.25, 0.10499356], [0.19661193, 0.04517666]]))

def test_relu():
    """Test the relu function
    """
    sig_value = relu(0)
    sig_value_2 = relu(np.array([0, 2]))
    sig_value_3 = relu(np.array([[0, 0.011], [1.12565, 3.76762]]))
    assert sig_value == 0
    assert np.array_equal(sig_value_2, np.array([0, 2]))
    assert np.array_equal(sig_value_3, np.array([[0, 0.011], [1.12565, 3.76762]]))

def test_derivative_relu():
    """Test the derivative of the relu function
    """
    ds_value = relu_derivative(0)
    ds_value_2 = relu_derivative(np.array([0, 2]))
    ds_value_3 = relu_derivative(np.array([[0, 0.011], [1.12565, 3.76762]]))
    assert ds_value == 0
    assert np.allclose(ds_value_2, np.array([0, 1]))
    assert np.allclose(ds_value_3, np.array([[0, 1], [1, 1]]))

def test_creation_of_activation_function():
    """Test the creation of activation function and the raise ValueError
    """
    activation = ActivationFunction(name="sigmoid")
    activation_authorized = ["sigmoid", "relu"]
    val = activation.get_function_returns(np.array([[0, 2], [1, 3]]))
    d_val = activation.get_derivative_function_returns(np.array([0, 2]))
    print(d_val)
    assert activation.get_name() == "sigmoid"
    with pytest.raises(
        ValueError,
        match=re.escape(f"Invalid activation function, expected one of {activation_authorized}")
    ):
        ActivationFunction(name="it's me")
    assert np.allclose(val, np.array([[0.5, 0.88079708], [0.73105858, 0.95257413]]))
    assert np.allclose(d_val, np.array([0.25, 0.10499356]))
