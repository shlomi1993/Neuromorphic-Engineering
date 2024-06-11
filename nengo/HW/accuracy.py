import nengo
import numpy as np
import matplotlib.pyplot as plt

tau_synapse = 0.1

# Model definition
model = nengo.Network()
with model:
    stim = nengo.Node(lambda t: np.sin(t))
    ens1 = nengo.Ensemble(n_neurons=1, dimensions=1)  # To attempt to represent simple sine using a single neuron    
    ens2 = nengo.Ensemble(n_neurons=2, dimensions=1)  # To attempt to represent simple sine using two neurons
    ens3 = nengo.Ensemble(n_neurons=100, dimensions=1)  # To show the representation improvement as we add neurons
    nengo.Connection(stim, ens1)
    nengo.Connection(stim, ens2)
    nengo.Connection(stim, ens3)

    # Probes to record data
    probe_stim = nengo.Probe(stim)
    probe1 = nengo.Probe(ens1, synapse=tau_synapse)
    probe2 = nengo.Probe(ens2, synapse=tau_synapse)
    probe3 = nengo.Probe(ens3, synapse=tau_synapse)

# Model simulation
with nengo.Simulator(model) as sim:
    sim.run(15.0)

# Plot results
t = sim.trange()
plt.figure(figsize=(12, 11))
plt.subplot(2, 2, 1)
plt.title('Input Signal')
plt.plot(t, sim.data[probe_stim])
plt.xlabel('Time (s)')
plt.ylabel('Input')
plt.subplot(2, 2, 2)
plt.title('Single Neuron Representation')
plt.plot(t, sim.data[probe1], color='red')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.subplot(2, 2, 3)
plt.title('Two Neurons Representation')
plt.plot(t, sim.data[probe2], color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.subplot(2, 2, 4)
plt.title('100 Neuron Representation')
plt.plot(t, sim.data[probe3], color='green')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.show()
