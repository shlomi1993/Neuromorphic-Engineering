import nengo

model = nengo.Network()
with model:
	a = nengo.Ensemble(
		n_neurons=2,
		dimensions=1,
		encoders=[[1], [-1]],
		max_rates=[100, 100],
		intercept=[-0.5, -0.5],
		neuron_type=nengo.LIF()
	)

	# stim = nengo.Node(0)
	stim = nengo.Node(1)
	# stim can be controlled using a GUI slider

	nengo.Connection(stim, a)