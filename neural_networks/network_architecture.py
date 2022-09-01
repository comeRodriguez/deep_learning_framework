"""This file is for testing the combinations of different layers
"""
from layer import FullyConnected
from utils.utils import load_planar_dataset

FEATURES, LABELS = load_planar_dataset(n_example=10)
print(FEATURES)
print(LABELS)
first_layer = FullyConnected(input_feature=FEATURES.T, number_of_neurons=4, activation_function="relu")
first_layer.compute_activations()
second_layer = FullyConnected(input_feature=first_layer.activations, number_of_neurons=1, activation_function="sigmoid")
second_layer.compute_activations()
print(first_layer.activations)
print()
print(second_layer.activations)
