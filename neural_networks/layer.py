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
    
    def compute_derivatives(
        self,
        weights_foll_layer: np.ndarray,
        d_linear_part_foll_layer: np.ndarray,
        prev_activations: np.ndarray,
        n_examples: int
        ) -> None:
        self.derivative_activations = np.dot(weights_foll_layer.T, d_linear_part_foll_layer)
        self.derivative_linear = self.derivative_activations *  \
            self.activation_function.get_derivative_function_returns(self.linear_part)
        self.derivative_weights = 1/n_examples * np.dot(self.derivative_linear, prev_activations.T)
        self.derivative_biais = 1/n_examples * np.sum(self.derivative_linear, axis=1, keepdims=True)
    
    def compute_derivatives_last_layer(self, prev_activations, n_examples: int, cost_derivative) -> None:
        self.derivative_activations = cost_derivative
        self.derivative_linear = self.derivative_activations *  \
            self.activation_function.get_derivative_function_returns(self.linear_part)
        # self.derivative_linear = self.activations - y_true
        self.derivative_weights = 1/n_examples * np.dot(self.derivative_linear, prev_activations.T)
        self.derivative_biais = 1/n_examples * np.sum(self.derivative_linear, axis=1, keepdims=True)

    def update_parameters(self, learning_rate: float) -> None:
        self.weights = self.weights - learning_rate*self.derivative_weights
        self.biais = self.biais - learning_rate*self.derivative_biais

