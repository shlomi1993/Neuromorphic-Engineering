# from nengo import Network, Node, Ensemble, Connection

# model = Network()
# with model:
# 	a = Ensemble(n_neurons=70, dimenstions=1, radius=1)
# 	stim = Node(0)
# 	Connection(stim, a)

# 	b = Ensemble(n_neurons=30, dimenstions=1)
# 	Connection(a, b)



from nengo import Network, Node, Ensemble, Connection

model = Network()
with model:
	a = Ensemble(n_neurons=70, dimenstions=2, radius=1)
	stim = Node([0, 0])
	Connection(stim, a)

	b = Ensemble(n_neurons=30, dimenstions=1)

	# Use data instead of a function
	inputs = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
	outputs = [[1], [-1], [-1], [1]]

	Connection(a, b, eval_points=inputs, function=outputs)
