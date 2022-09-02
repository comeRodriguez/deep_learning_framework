"""This file contains all the activation functions and their derivatives
of this personal framework
"""
from typing import Tuple
import numpy as np


def sigmoid(arg: Tuple[np.ndarray, float]) -> Tuple[np.ndarray, float]:
    """Compute the sigmoid function of z

    Args:
        arg (Tuple[np.array, float]): scalar or numpy array.
            Arguments of the function (i.e x in f(x))

    Returns:
        Tuple[np.array, float]: computed sigmoid(arg)
    """
    sig = 1/(1+np.exp(-arg))
    return sig

def sigmoid_derivative(arg: Tuple[np.ndarray, float]) -> Tuple[np.ndarray, float]:
    """Derivative of the sigmoid function of arg: d(sigmoid)/d_arg

    Args:
        arg (Tuple[np.ndarray, float]): scalar or numpy array.
            Arguments of the function (i.e x in f(x))

    Returns:
        Tuple[np.ndarray, float]: computed derivative d(sigmoid(arg))
    """
    d_sig = sigmoid(arg)*(1 - sigmoid(arg))
    return d_sig

def relu_func(arg: Tuple[np.ndarray, float]) -> Tuple[np.ndarray, float]:
    """
    Compute the relu of arg

    Args:
        arg (Tuple[np.array, float]): scalar or numpy array.
            Arguments of the function (i.e x in f(x))

    Returns:
        Tuple[np.array, float]: computed relu(arg)
    """
    relu = np.maximum(0, arg)
    return relu

def relu_derivative(arg: Tuple[np.ndarray, float]) -> Tuple[np.ndarray, float]:
    """Derivative of the relu function of arg: d(relu)/d_arg

    Args:
        arg (Tuple[np.ndarray, float]): scalar or numpy array.
            Arguments of the function (i.e x in f(x))

    Returns:
        Tuple[np.ndarray, float]: computed derivative d(relu(arg))
    """
    d_relu = (arg > 0) * 1
    return d_relu

class ActivationFunction():
    """Activation function class
    """
    def __init__(self, name: str) -> None:
        """Constructor of the class

        Args:
            name (str): name of the activation function to use.
                Choice between [sigmoid, relu]

        Raises:
            ValueError: raise if the name is not one of the available function
        """
        authorized_names = [
            "sigmoid",
            "relu"
        ]
        if name not in authorized_names:
            raise ValueError(f"Invalid activation function, expected one of {authorized_names}")
        self.name = name
        self.corresponding_functions = {
            "sigmoid": [sigmoid, sigmoid_derivative],
            "relu": [relu_func, relu_derivative]
        }

    def get_name(self) -> str:
        """Return the name of the activation function

        Returns:
            str: name of thz activation function
        """
        return self.name

    def get_function_returns(self, arg: Tuple[np.ndarray, float]) -> Tuple[np.ndarray, float]:
        """Get the results of the activation function of arg

        Args:
            arg (Tuple[np.ndarray, float]): scalar or numpy array.
                Arguments of the function (i.e x in f(x))

        Returns:
            Tuple[np.ndarray, float]: return of activation_function(arg)
        """
        val = self.corresponding_functions[self.name][0](arg)
        return val

    def get_derivative_function_returns(
        self,
        arg: Tuple[np.ndarray, float]
    ) -> Tuple[np.ndarray, float]:
        """Get the results of the derivative of activation function of arg

        Args:
            arg (Tuple[np.ndarray, float]): scalar or numpy array.
                Arguments of the function (i.e x in f(x))

        Returns:
            Tuple[np.ndarray, float]: return of d(activation_function(arg))
        """
        val = self.corresponding_functions[self.name][1](arg)
        return val
