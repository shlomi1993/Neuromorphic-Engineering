import nengo
import matplotlib.pyplot as plt

tau_synapse = 0.1

# Model definition
model = nengo.Network()
with model:
    stim = nengo.Node([0.5, -0.5])
    ens1 = nengo.Ensemble(n_neurons=1, dimensions=2)  # To attempt to represent 2D input using a single neuron    
    ens2 = nengo.Ensemble(n_neurons=5, dimensions=2)  # To attempt to represent 2D input using sufficient neurons
    nengo.Connection(stim, ens1)
    nengo.Connection(stim, ens2)
    
    probe_stim = nengo.Probe(stim, synapse=tau_synapse)
    probe1 = nengo.Probe(ens1, synapse=tau_synapse)
    probe2 = nengo.Probe(ens2, synapse=tau_synapse)

# Model simulation
with nengo.Simulator(model) as sim:
    sim.run(1.0)

# Plot results
t = sim.trange()
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Single Neuron Representation')
plt.plot(t, sim.data[probe_stim], '--', color='red', label='Stimulus')
plt.plot(t, sim.data[probe1])
plt.xlabel('Time (s)')
plt.ylabel('Representation')
plt.subplot(1, 2, 2)
plt.title('Five Neurons Representation')
plt.plot(t, sim.data[probe_stim], '--', color='red', label='Stimulus')
plt.plot(t, sim.data[probe2])
plt.xlabel('Time (s)')
plt.ylabel('Representation')
plt.show()
