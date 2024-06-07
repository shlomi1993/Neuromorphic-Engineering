import nengo

tau_desired = 0.05  # 0.5
tau_synapse = 0.005  # 0.1

def forward(u):
    return (tau_synapse / tau_desired) * u

def recurrent(x):
    # return tau_synapse * (-1 / tau_desired) * x + x
    return (1 - tau_synapse / tau_desired) * x  # Simplified

model = nengo.Network()
with model:
    a = nengo.Ensemble(n_neurons=100, dimensions=1)
    b = nengo.Ensemble(n_neurons=100, dimensions=1)
    stim = nengo.Node(0)
    nengo.Connection(stim, a)
    nengo.Connection(a, b, function=forward, synapse=tau_synapse)
    nengo.Connection(a, b, function=recurrent, synapse=tau_synapse)
