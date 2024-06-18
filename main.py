import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import os

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
raw_dataset = pd.read_csv("dataset.csv")
dataset = raw_dataset.copy().drop('Serial No.', axis=1)
dataset.tail()
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
sns.pairplot(train_dataset
    [['GRE', 'TOEFL', 'UniversityRating', 'SOP', 'LOR', 'CGPA', 'Research', 'ChanceOfAdmit']], 
    diag_kind='kde')
train_dataset.describe().transpose()
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('ChanceOfAdmit')
test_labels = test_features.pop('ChanceOfAdmit')
train_dataset.describe().transpose()[['mean', 'std']]
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())
first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])
linear_model.predict(train_features[:10])
linear_model.summary()
linear_model.layers[1].kernel
linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error',)
history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)
test_results = {}

test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels)
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [ChanceOfAdmit]')
  plt.legend()
  plt.grid(True)

def plot_predictions(x, y):
  p = sns.regplot(x=x, y=y)
  p.set(xlabel='Chance of Admit (actual)', ylabel='Chance of Admit (predicted)')
  slope, intercept, r, p, sterr = scipy.stats.linregress(x=p.get_lines()[0].get_xdata(),
                        y=p.get_lines()[0].get_ydata())
  print(f"y = {slope:.2f}x + {intercept:.2f}\nr={r:.2f}\nR^2 = {r**2:.2f}")
plot_predictions(test_labels, linear_model.predict(test_features))
weights = linear_model.layers[1].get_weights()[0]
bias = linear_model.layers[1].get_weights()[1]

print("Weights\n", weights)
print("Bias\n", bias)
pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).