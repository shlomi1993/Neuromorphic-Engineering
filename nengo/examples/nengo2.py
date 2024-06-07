import nengo
import numpy as np

model = nengo.Network()
with model:
	a = nengo.Ensemble(n_neurons=100, dimensions=1)
	stim = nengo.Node(np.sin)
	nengo.Connection(stim, a)

	# The 'None' is to let this node do nothing but letting us plot the connection output
	# And, this node has to 'know' the input dim so size_in=1
	output = nengo.Node(None, size_in=1)

	# Learn the outout from the neurons of 'a' using a weight matrix of zeros and PES learning rule with a defined eta
	c = nengo.Connection(a.neurons, output, transform=np.zeros(1, 100), learning_rule=nengo.PES(learning_rate=0.0001))

	# Assuming the error comes from the environment
	error = nengo.Node(None, size_in=1)

	# Learn the error from the output using the identity weight matrix
	nengo.Connection(output, error)
	nengo.Connection(stim, error, transform=-1)  # ?
	nengo.Connection(error, c.learning_rule)  # ?
