"""This file is for testing the model class
"""
from unittest import TestCase
import numpy as np
from neural_networks.model import Model
from neural_networks.layer import FullyConnected
from utils.utils import load_planar_dataset

FEATURES, LABELS = load_planar_dataset(n_example=10)

def test_model_creation():
    """Test initialization of the Model class
    """
    model = Model(input_layer=FEATURES.T)
    assert np.array_equal(model.input_layer, FEATURES.T)
    assert len(model.layers) == 1
    assert np.array_equal(FEATURES.T, model.layers[0])
    TestCase.assertIsNone(model.cost, model.cost)
    TestCase.assertIsNone(model.learning_rate, model.learning_rate)

def test_model_add_layer():
    """Test the Model.add_layer() method
    """
    model = Model(input_layer=FEATURES.T)
    model.add_layer(FullyConnected(number_of_neurons=10, activation_function="relu"))
    model.add_layer(FullyConnected(number_of_neurons=20, activation_function="relu"))
    assert len(model.layers) == 3
    assert model.layers[1].weights.shape == (10, 2)
    assert model.layers[2].weights.shape == (20, 10)
    assert model.layers[1].biais.shape == (10, 1)
    assert model.layers[2].biais.shape == (20, 1)

def test_model_build():
    """Test the Model.build_model()
    """
    model = Model(input_layer=FEATURES.T)
    model.build_model(cost_function="binary_cross_entropy", learning_rate=0.01)
    assert model.cost.get_name() == "binary_cross_entropy"
    assert model.learning_rate == 0.01
