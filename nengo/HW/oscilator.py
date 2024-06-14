import nengo
import matplotlib.pyplot as plt

tau_synapse = 0.01
r = -0.15  # Sets the oscillator frequency

# Recurrent function, defines the oscillator dynamic
def feedback(x):
    return x[0] + r * x[1], x[1] - r * x[0]

# Model definition
model = nengo.Network('Oscillator')
with model:
    input_node = nengo.Node(lambda t: [0.5, 0.5] if t < 0.02 else [0, 0])
    ensemble = nengo.Ensemble(n_neurons=200, dimensions=2)
    nengo.Connection(ensemble, ensemble, function=feedback, synapse=tau_synapse)
    nengo.Connection(input_node, ensemble)
    probe_input = nengo.Probe(input_node)
    probe_ensemble = nengo.Probe(ensemble, synapse=tau_synapse)

# Model simulation
with nengo.Simulator(model) as sim:
    sim.run(1.0)

# Plot results
t = sim.trange()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t, sim.data[probe_ensemble], label=['$x_0$', '$x_1$'])
plt.plot(t, sim.data[probe_input], 'r', label=['stim', ''], linewidth=2)
plt.xlabel('Time (s)', fontdict={'size': 14})
plt.ylabel('State Value', fontdict={'size': 14})
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(sim.data[probe_ensemble][:,0],sim.data[probe_ensemble][:,1])
plt.xlabel('$x_0$', fontdict={'size': 14})
plt.ylabel('$x_1$', fontdict={'size': 14})
plt.tight_layout()
plt.show()
