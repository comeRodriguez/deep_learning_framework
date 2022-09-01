"""This file contains several cost functions
"""
import numpy as np

def binary_cross_entropy(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """Compute the binary cross entropy of a set of predicted labels and a set of true labels.
    The dimensions of true_labels and predicted_labels have to be equal.
    (formula: 
        L = true_labels*log(predicted_label) + (1-true_labels)*log(1-predicted_labels)
        BCE = -1/m * sum(L) for all labels in true_labels and predicted_labels
    )

    Args:
        true_labels (np.ndarray): vector of true labels for the supervised algorithm.
            The labels are 0 or 1.
        predicted_labels (np.ndarray): predicted labels as the output of a neural network.
            The labels have to be float between 0 and 1

    Returns:
        float: Computed binary cross entropy loss
    """
    n_labels = true_labels.shape[0]
    binary_cross_entropy = -(1/n_labels)*np.sum(
        true_labels*np.log(predicted_labels) + (1-true_labels)*np.log(1-predicted_labels)
    )
    return binary_cross_entropy

def binary_cross_entropy_derivative(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray
) -> float:
    """Compute the derivative of binary cross entropy of a set of predicted labels and
    a set of true labels according to the predicted labels.
    The dimensions of true_labels and predicted_labels have to be equal.
    (formula: 
        L = true_labels*log(predicted_label) + (1-true_labels)*log(1-predicted_labels)
        BCE = -1/m * sum(L) for all labels in true_labels and predicted_labels

        DERIVATIVE:
        dL/d(predicted_labels) = true_labels/predicted - (1-true_labels)/(1-predicted_labels)
        d(BCE)/d(predicted_labels) = -1/m * sum(dL/d(predicted_labels))
    )

    Args:
        true_labels (np.ndarray): vector of true labels for the supervised algorithm.
            The labels are 0 or 1.
        predicted_labels (np.ndarray): predicted labels as the output of a neural network.
            The labels have to be float between 0 and 1

    Returns:
        float: Computed derivative of the binary cross entropy loss
    """
    n_labels = true_labels.shape[0]
    d_binary_cross_entropy = -(1/n_labels) * np.sum(
        true_labels/predicted_labels - (1-true_labels)/(1-predicted_labels)
    )
    return d_binary_cross_entropy

class CostFunction():
    """Cost function class
    """
    def __init__(self, name: str) -> None:
        """Initialization of the class

        Args:
            name (str): name of the cost function wanted to be use.
                Choice between [binary_cross_entropy,]

        Raises:
            ValueError: raise if the name is not one of the available function
        """
        authorized_names = [
            "binary_cross_entropy",
        ]
        if name not in authorized_names:
            raise ValueError(f"Invalid activation function, expected one of {authorized_names}")
        self.name = name
        self.corresponding_functions = {
            "binary_cross_entropy": [binary_cross_entropy, binary_cross_entropy_derivative],
        }

    def get_name(self) -> str:
        """Return the name of the activation function

        Returns:
            str: name of thz activation function
        """
        return self.name

    def get_computed_cost(self, true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
        """Get the results of the cost function between true_labels and
        predicted_labels

        Args:
            true_labels (np.ndarray): vector of true labels for the supervised algorithm.
                The labels are 0 or 1.
            predicted_labels (np.ndarray): predicted labels as the output of a neural network.
                The labels have to be float between 0 and 1

        Returns:
            Tuple[np.ndarray, float]: return of activation_function(arg)
        """
        cost = self.corresponding_functions[self.name][0](
            true_labels=true_labels,
            predicted_labels=predicted_labels
        )
        return cost

    def get_derivative_computed_cost(
        self,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray
    ) -> float:
        """Get the results of the derivative of cost function

        Args:
            true_labels (np.ndarray): vector of true labels for the supervised algorithm.
                The labels are 0 or 1.
            predicted_labels (np.ndarray): predicted labels as the output of a neural network.
                The labels have to be float between 0 and 1

        Returns:
            float: computed derivative
        """
        d_cost = self.corresponding_functions[self.name][1](
            true_labels=true_labels, predicted_labels=predicted_labels
        )
        return d_cost
