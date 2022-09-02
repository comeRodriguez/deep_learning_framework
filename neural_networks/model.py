"""This file contains neural network model implementation
"""
import numpy as np
from neural_networks.layer import FullyConnected
from neural_networks.cost import CostFunction

class Model():
    """Model class
    """
    def __init__(self, input_layer: np.ndarray) -> None:
        """Initialization of the class

        Args:
            input_layer (np.ndarray): Model input
        """
        self.input_layer = input_layer
        self.layers = [self.input_layer]
        self.cost: CostFunction = None
        self.learning_rate: float = None

    def add_layer(self, layer: FullyConnected) -> None:
        """Add a layer into the neural network model

        Args:
            layer (FullyConnected): Layer to add
        """
        if len(self.layers) == 1:
            layer.update_input(self.layers[-1])
            layer.initialize_weights_and_biais(n_prev_units=self.input_layer.shape[0])
        else:
            layer.initialize_weights_and_biais(n_prev_units=self.layers[-1].units)
        self.layers.append(layer)

    def summary(self) -> None:
        """Get a summary of the constructed neural network model
        """
        for index, layer in enumerate(self.layers):
            if index == 0:
                print("--------- INPUT ---------")
                print(f"shape: {self.input_layer.shape}")
                print("-------------------------")
            else:
                print(f"--------- LAYER {index} ---------")
                print(f"number of neurons: {layer.units}")
                print(f"number of weights: {layer.weights.shape}")
                print(f"activation function: {layer.activation_function.get_name()}")
                print("---------------------------")
        if self.cost is not None and self.learning_rate is not None:
            print("--------- COMPILATION ---------")
            print(f"cost function: {self.cost.get_name()}")
            print(f"learning rate: {self.learning_rate}")
            print("-------------------------------")

    def build_model(self, cost_function: str, learning_rate: float):
        """Finalize the construction of the model by adding the cost function
        and the learning rate for the gradient descent

        Args:
            cost_function (str): cost function to optimize
            learning_rate (float): learning rate used for gtadient descent
        """
        self.cost = CostFunction(name=cost_function)
        self.learning_rate = learning_rate
