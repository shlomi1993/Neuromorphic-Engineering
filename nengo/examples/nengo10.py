import nengo

# dx/dt = u(t)

tau_synapse = 0.02  # 0.005

def forward(u):
    return tau_synapse * u

def recurrent(x):
    return x

model = nengo.Network()
with model:
    a = nengo.Ensemble(n_neurons=100, dimensions=1)
    b = nengo.Ensemble(n_neurons=100, dimensions=1)
    stim = nengo.Node(0)
    nengo.Connection(stim, a)
    nengo.Connection(a, b, function=forward, synapse=tau_synapse)  # Same as: nengo.Connection(a, b, transform=tau_synapse, synapse=tau_synapse)
    nengo.Connection(b, b, function=recurrent, synapse=tau_synapse)  # Same as: nengo.Connection(b, b, synapse=tau_synapse)
