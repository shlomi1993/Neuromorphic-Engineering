import nengo
import numpy as np
import matplotlib.pyplot as plt

tau_synapse = 0.01
nonlinear = lambda x: x ** 2

# Model definition
model = nengo.Network()
with model:
    stim = nengo.Node(lambda t: np.sin(t))
    ens1 = nengo.Ensemble(n_neurons=1, dimensions=1)  # To attempt to represent f(x)=x^2 using a single neuron    
    ens2 = nengo.Ensemble(n_neurons=50, dimensions=1)  # To attempt to represent f(x)=x^2 using sufficient neurons
    nengo.Connection(stim, ens1)
    nengo.Connection(stim, ens2)
    
    # Connecting ensembles to output nodes with nonlinear function
    out1 = nengo.Node(size_in=1)
    out2 = nengo.Node(size_in=1)
    nengo.Connection(ens1, out1, function=nonlinear)
    nengo.Connection(ens2, out2, function=nonlinear)
    
    # Probes to record data
    probe_stim = nengo.Probe(stim)
    probe1 = nengo.Probe(out1, synapse=tau_synapse)
    probe2 = nengo.Probe(out2, synapse=tau_synapse)

# Model simulation
with nengo.Simulator(model) as sim:
    sim.run(20.0)

# Plot results
t = sim.trange()
plt.figure(figsize=(12, 9))
plt.subplot(2, 1, 1)
plt.title('Input Signal')
plt.plot(t, sim.data[probe_stim])
plt.ylabel('Input')
plt.subplot(2, 1, 2)
plt.title('Nonlinear Function Representation')
plt.plot(t, sim.data[probe1], label='1 Neuron')
plt.plot(t, sim.data[probe2], label='50 Neurons')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.legend()
plt.show()
