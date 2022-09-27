"""This file contains neural network model implementation
"""
from typing import List, Union
import numpy as np
from neural_networks.layer import FullyConnected
from neural_networks.cost import CostFunction
from sklearn.metrics import accuracy_score
import pandas as pd

class Model():
    """Model class
    """
    def __init__(self, input_layer: np.ndarray) -> None:
        """Initialization of the class

        Args:
            input_layer (np.ndarray): Model input
        """
        self.input_layer = input_layer
        self.layers: List[Union[np.ndarray, FullyConnected]] = [self.input_layer]
        self.cost: CostFunction = None
        self.costs = []
        self.learning_rate: float = None
        self.accuracies: List = [] 

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

    def build_model(self, cost_function: str, learning_rate: float) -> None:
        """Finalize the construction of the model by adding the cost function
        and the learning rate for the gradient descent

        Args:
            cost_function (str): cost function to optimize
            learning_rate (float): learning rate used for gtadient descent
        """
        self.cost = CostFunction(name=cost_function)
        self.learning_rate = learning_rate
    
    def forward_propagation(self) -> None:
        """Implementation of the forward propagation in a neural network
        """
        for index, layer in enumerate(self.layers[1:]):
            if index > 0:
                layer.update_input(self.layers[index].activations)
            layer.compute_activations()
    
    def get_cost(self, y_true: np.ndarray) -> float:
        """Get cost value with the actual activations of the model's last layer

        Args:
            y_true (np.ndarray): true labels used to compute the cost function

        Returns:
            float: value of the cost function
        """
        self.forward_propagation()
        cost_value = self.cost.get_computed_cost(
            true_labels=y_true, predicted_probas=self.layers[-1].activations
        )
        return cost_value
    
    def backward_propagation(self, y_true: np.ndarray) -> None:
        """Compute the gradient descent algorithm on all the network

        Args:
            y_true (np.ndarray): true labels
        """
        d_cost = self.cost.get_derivative_computed_cost(
            true_labels=y_true, predicted_probas=self.layers[-1].activations
            )
        for index in reversed(range(1, len(self.layers))):
            if index == len(self.layers) - 1 and len(self.layers) > 2:
                self.layers[index].compute_derivatives_last_layer(
                    cost_derivative=d_cost,
                    prev_activations=self.layers[index-1].activations,
                    n_examples=len(y_true),
                )
            elif index == 1 and len(self.layers) > 2:
                self.layers[index].compute_derivatives(
                    weights_foll_layer=self.layers[index+1].weights,
                    d_linear_part_foll_layer=self.layers[index+1].derivative_linear,
                    prev_activations=self.layers[0],
                    n_examples=len(y_true)
                )
            elif len(self.layers) == 2:
                self.layers[index].compute_derivatives_last_layer(
                    cost_derivative=d_cost,
                    prev_activations=self.layers[0],
                    n_examples=len(y_true),
                )
            else:
                self.layers[index].compute_derivatives(
                    weights_foll_layer=self.layers[index+1].weights,
                    d_linear_part_foll_layer=self.layers[index+1].derivative_linear,
                    prev_activations=self.layers[index-1].activations,
                    n_examples=len(y_true)
                )
            self.layers[index].update_parameters(learning_rate=self.learning_rate)

    def fit(self, y_true: np.ndarray, epochs: int):
        """Fit the model with the given labels

        Args:
            y_true (np.ndarray): true labels
            epochs (int): number of iterations
        """
        for _ in range(epochs):
            self.forward_propagation()
            hypotetic_labels = pd.Series(self.layers[-1].activations[0,:])
            hypotetic_labels = hypotetic_labels.apply(lambda x: 1 if x >= 0.5 else 0)
            self.accuracies.append(accuracy_score(y_true, hypotetic_labels))
            self.costs.append(self.get_cost(y_true=y_true))
            self.backward_propagation(y_true=y_true)
