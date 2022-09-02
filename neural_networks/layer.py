"""This file contains neural network layers implementation
"""
import numpy as np
from utils.utils import initialize_randomly
from neural_networks.activation_functions import ActivationFunction

class FullyConnected():
    """Fully connected layer in a neural network
    """
    def __init__(
        self,
        number_of_neurons: int,
        activation_function: str,
        random_seed: int=None
    ) -> None:
        """Initialization of the class

        Args:
            number_of_neurons (int): number of neurons wanted to be present in the layer
            activation_function (ActivationFunction): activation function to compute the
                activation values
            random_seed (int, optional): If present, this will allow the user to compute
                always the same weights. Defaults to None.
        """
        self.units = number_of_neurons
        self.random_seed = random_seed
        self.input = None
        self.weights: np.ndarray = None
        self.biais: np.ndarray = None
        self.activation_function = ActivationFunction(name=activation_function)
        self.linear_part: np.ndarray = None
        self.activations:np.ndarray = None
        self.derivative_activations: np.ndarray = None
        self.derivative_linear: np.ndarray = None
        self.derivative_weights: np.ndarray = None
        self.derivative_biais: np.ndarray = None

    def update_input(self, input_layer: np.ndarray) -> None:
        """_summary_
        """
        self.input = input_layer

    def initialize_weights_and_biais(self, n_prev_units:int) -> None:
        """_summary_
        """
        self.weights, self.biais = initialize_randomly(
            input_dim=n_prev_units,
            n_units=self.units,
            random_seed=self.random_seed
        )


    def compute_linear_part(self):
        """Compute the linear part of a neuron in a neural network
        (i.e => compute Z = Weights.Features + Biais)
        """
        self.linear_part = np.dot(self.weights, self.input) + self.biais

    def compute_activations(self):
        """Compute the activation values of a neuron in a neural network
        using the activation function g
        (i.e => A = g(Z))
        """
        self.compute_linear_part()
        self.activations = self.activation_function.get_function_returns(arg=self.linear_part)
