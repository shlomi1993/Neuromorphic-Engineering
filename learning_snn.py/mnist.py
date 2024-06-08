import nengo
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import nengo_dl

from urllib.request import urlretrieve


# MNIST classification - source: https://www.nengo.ai/nengo-dl/examples/spiking-mnist.html

# Data loading and image flattening
(train_images, train_labels), (test_images, test_labels,) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((train_images.shape[0], -1))
test_images = test_images.reshape((test_images.shape[0], -1))

# Model definition
with nengo.Network(seed=0) as net:
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None
    neuron_type = nengo.LIF(amplitude=0.01)

    # This is an optimization to improve the training speed, since we won't require stateful behavior in this example
    nengo_dl.configure_settings(stateful=False)

    # The input node 
    inp = nengo.Node(np.zeros(28 * 28))

    # Add the first convolutional layer
    x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=32, kernel_size=3))(inp, shape_in=(28, 28, 1))
    x = nengo_dl.Layer(neuron_type)(x)

    # Add the second convolutional layer
    x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=64, strides=2, kernel_size=3))(x, shape_in=(26, 26, 32))
    x = nengo_dl.Layer(neuron_type)(x)

    # Add the third convolutional layer
    x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=128, strides=2, kernel_size=3))(x, shape_in=(12, 12, 64))
    x = nengo_dl.Layer(neuron_type)(x)

    # Linear readout
    out = nengo_dl.Layer(tf.keras.layers.Dense(units=10))(x)

    # Probes
    out_p = nengo.Probe(out, label='out_p')
    out_p_filt = nengo.Probe(out, synapse=0.1, label='out_p_filt')

# Network building
minibatch_size = 200
sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size)

# Pre-processing

## Add single timestep to training data
train_images = train_images[:, None, :]
train_labels = train_labels[:, None, None]

## When testing our network with spiking neurons we will need to run it over time, so we repeat the input/target data 
## for a number of timesteps.
n_steps = 30
test_images = np.tile(test_images[:, None, :], (1, n_steps, 1))
test_labels = np.tile(test_labels[:, None, None], (1, n_steps, 1))

# SNN compilation before training
def classification_accuracy(y_true, y_pred):
    return tf.metrics.sparse_categorical_accuracy(y_true[:, -1], y_pred[:, -1])

## Note that we use `out_p_filt` when testing (to reduce the spike noise)
sim.compile(loss={out_p_filt: classification_accuracy})
print("Accuracy before training:", sim.evaluate(test_images, {out_p_filt: test_labels}, verbose=0)["loss"],)
## PRINTED: Accuracy before training: 0.09340000152587890:00:00

# Training
do_training = False
if do_training:
    sim.compile(optimizer=tf.optimizers.RMSprop(0.001), loss={out_p: tf.losses.SparseCategoricalCrossentropy(from_logits=True)},)
    sim.fit(train_images, {out_p: train_labels}, epochs=10)
    sim.save_params("./mnist_params")  # save the parameters to file
else:
    urlretrieve("https://drive.google.com/uc?export=download&id=1l5aivQljFoXzPP5JVccdFXbOYRv3BCJR", "mnist_params.npz")
    sim.load_params("./mnist_params")

# Evaluating after training
sim.compile(loss={out_p_filt: classification_accuracy})
print("Accuracy after training:", sim.evaluate(test_images, {out_p_filt: test_labels}, verbose=0)["loss"])
## PRINTED: Accuracy after training: 0.9869999885559082 0:00:00 

# Plot
data = sim.predict(test_images[:minibatch_size])
for i in range(5):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(test_images[i, 0].reshape((28, 28)), cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.plot(tf.nn.softmax(data[out_p_filt][i]))
    plt.legend([str(i) for i in range(10)], loc="upper left")
    plt.xlabel("timesteps")
    plt.ylabel("probability")
    plt.tight_layout()
