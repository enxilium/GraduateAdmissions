import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

new_model = tf.keras.models.load_model('model.keras')



raw_dataset = pd.read_csv("dataset.csv")

dataset = raw_dataset.copy().drop('Serial No.', axis=1)
dataset.tail()

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('ChanceOfAdmit')
test_labels = test_features.pop('ChanceOfAdmit')

print(new_model.summary())

weights = new_model.layers[1].get_weights()[0]
bias = new_model.layers[1].get_weights()[1]

print("Weights\n", weights)
print("Bias\n", bias)

test = np.array([[337, 118, 4, 4.5, 4.5, 9.65, 1]])
test2 = np.array([[0, 0, 0, 0, 0, 0, 0]])
print(new_model.predict(test))
print(new_model.predict(test2))