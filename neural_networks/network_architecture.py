"""This file is for testing the combinations of different layers
"""
from neural_networks.layer import FullyConnected
from neural_networks.model import Model
from utils.utils import load_planar_dataset

FEATURES, LABELS = load_planar_dataset(n_example=10)
model = Model(input_layer=FEATURES.T)
model.add_layer(FullyConnected(number_of_neurons=4, activation_function="relu"))
model.add_layer(FullyConnected(number_of_neurons=4, activation_function="relu"))
model.add_layer(FullyConnected(number_of_neurons=1, activation_function="sigmoid"))
model.build_model(cost_function="binary_cross_entropy", learning_rate=0.01)
model.summary()
