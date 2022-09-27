"""This file is for testing the combinations of different layers
"""
from neural_networks.layer import FullyConnected
from neural_networks.model import Model
import pandas as pd
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras import layers

df = pd.read_csv("mock_data/SynthPara_n1000_p2.csv")
FEATURES = np.round(df[["X1", "X2"]].to_numpy(), decimals=2)
LABELS = df["z"].apply(lambda x: 0 if x=="A" else 1).to_numpy()
model = Model(input_layer=FEATURES.T)
model.add_layer(FullyConnected(number_of_neurons=1, activation_function="sigmoid", random_seed=2))
model.build_model(cost_function="binary_cross_entropy", learning_rate=0.1)
model.summary()
model.fit(LABELS, epochs=200)
print()
print("ACCURACY")
print(model.accuracies[-1])
print("LOSS")
print(model.costs[-1])
print()
print("COMPARE WITH TENSORFLOW MODEL")
inpts = keras.Input(shape=(2,), name="input_layer")
outputs = layers.Dense(units=1, activation="sigmoid", name="outputs")(inpts)
lr = keras.Model(inputs=inpts, outputs=outputs, name="first_test")
lr.summary()
lr.compile(loss=BinaryCrossentropy(), metrics=["accuracy"])
history = lr.fit(FEATURES, LABELS, epochs=200, verbose=False)
history = pd.DataFrame(history.history).reset_index()
print(history.iloc[-1])
