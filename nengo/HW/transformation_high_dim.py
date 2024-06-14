import nengo
import numpy as np
import matplotlib.pyplot as plt


tau_synapse = 0.01

# Definition of sin(x * y)
def sin_xy(xy):
    return np.sin(xy[0] * xy[1])


# Model definition
model = nengo.Network()
with model:

    # Create nodes and ensembles
    input_x = nengo.Node(lambda x: 0.5 * np.sin(10 * x))
    input_y = nengo.Node(lambda y: 0.5 * np.cos(10 * y))
    output = nengo.Ensemble(n_neurons=100, dimensions=1)
    ensemble = nengo.Ensemble(n_neurons=1000, dimensions=2)

    # Connect input nodes to the main ensemble, and the main ensemble to the output ensemble
    nengo.Connection(input_x, ensemble[0])
    nengo.Connection(input_y, ensemble[1])
    nengo.Connection(ensemble, output, function=sin_xy)

    # Create relevant probes
    probe_input_x = nengo.Probe(input_x, synapse=tau_synapse)
    probe_input_y = nengo.Probe(input_y, synapse=tau_synapse)
    probe_output = nengo.Probe(output, synapse=tau_synapse)


# Model simulation
with nengo.Simulator(model) as sim:
    sim.run(1.0)


# Plot results
t = sim.trange()
plt.figure()
plt.figure(figsize=(8, 5))
plt.title('Transformation from $x$ and $y$ to $sin(x \cdot y)$')
plt.plot(t, sim.data[probe_input_x], label='$x$')
plt.plot(t, sim.data[probe_input_y], label='$y$')
plt.plot(t, sim.data[probe_output], label='$sin(x \cdot y)$')
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.legend()
plt.show()
# plt.savefig('sin_xy.png')
