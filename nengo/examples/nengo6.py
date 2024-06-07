from nengo import Network, Node, Ensemble, Connection, Direct
from typing import List

Vector = List[float]

model = Network()
with model:
	stim = Node([0, 0])
	a = Ensemble(n_neurons=100, dimenstions=2, radius=1.5)
	Connection(stim, a)

	b = Ensemble(n_neurons=50, dimenstions=1)

	def product(x: Vector) -> float:
		return x[0] * x[1]

	Connection(a, b, function=product)


# model = Network()
# with model:
# 	stim = Node([0, 0])
# 	a = Ensemble(n_neurons=100, dimenstions=2, radius=1.5, neuron_type=Direct())  # Debbuging neuron type
# 	Connection(stim, a)

# 	b = Ensemble(n_neurons=50, dimenstions=1)

# 	def custom_func(x: Vector) -> float:
# 		return np.sin(x[0]) * np.sin(x[1])

# 	Connection(a, b, function=custom_func)


