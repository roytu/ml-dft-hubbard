
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D

# Load datasets
W = 2.5
E_gnds = np.load(f"E_gnds_{W}.npy")
n_gnds = np.load(f"n_gnds_{W}.npy")
vs = np.load(f"vs_{W}.npy")

# Define a model
"""
Having tested a few ML models, we have opted for a
convolutional neural network, which is found to perform
better than standard neural networks [27]. All the machine
learning algorithms have been implemented by using the
KERAS PYTHON package [28].

In the convolutional neural
network, in order to fully capture all the information, we
extend each of the occupation vectors by adding their first
k − 1 components to the end of the vector, thus creating
an (L + k − 1)-component-long vector. Here k is the size of
the one-dimensional convolution window, in our particular
case k = 3. Since the convolutional neural network slides the
kernel window over the feature vector by choosing k elements
at the time, such a prescription guarantees that there are kernel
windows, which contain both the first (k − 1) elements and the
last one of the on-site occupation. The convolutional neural
network used has eight convolutional filters, followed by two
fully connected layers each with 128 units, and finally an out-
put layer. The loss function used to construct the convolutional
neural network is the mean-squared error, and the optimizer is
the Adam algorithm.

Conv1D(8, (3,))
"""
# Input: n_gnds (16)
# Output: E_gnds (1)

# Take the first 8 values only and expand to L+k-1 to
# reflect periodicity

xs = n_gnds[:, :8]  # Use only spin-ups (should be the same as spin-downs)
xs = np.hstack([xs, xs[:, :2]])
xs = np.expand_dims(xs, axis=2)  # Reshape to fit with Conv1D
ys = E_gnds

model = Sequential()
model.add(Conv1D(8, (3,), input_shape=(10, 1)))
model.add(Activation("relu"))
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dense(1))

model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mean_squared_error"])
model.fit(xs, ys, epochs=3)

