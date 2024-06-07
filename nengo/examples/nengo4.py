from nengo import Network, Node, Ensemble, Connection

model = Network()
with model:
	a = Ensemble(n_neurons=50, dimenstions=1, radius=10)
	stim = Node(0)
	Connection(stim, a)

	output = Node(None, size_in=1)

	# def square(x):
	# 	return x ** 2

	# Connection(a, output, function=square, synapse=0.005)

	def sign(x):
		return -1 if x < x else 1

	Connection(a, output, function=sign, synapse=0.005)
