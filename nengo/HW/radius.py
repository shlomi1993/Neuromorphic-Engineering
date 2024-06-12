import nengo
import matplotlib.pyplot as plt

r1 = 1.0
r2 = 1.5

def split_to_axis(data_points):
    """
    Transforms [[x0, y0,], [x1, y1], ..., [xn, yn]] to [[x0, x1, ..., xn], [y0, y1, ..., yn]]
    """
    return list(zip(*data_points))

# Model definition
model = nengo.Network()
with model:
    stim = nengo.Node([1, 1])
    a = nengo.Ensemble(n_neurons=100, dimensions=2, radius=r1)
    b = nengo.Ensemble(n_neurons=100, dimensions=2, radius=r2)
    nengo.Connection(stim, a)
    nengo.Connection(stim, b)

    probe_a = nengo.Probe(a, synapse=0.01)
    probe_b = nengo.Probe(b, synapse=0.01)

# Model simulation
with nengo.Simulator(model) as sim:
    sim.run(1.0)

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.grid(True)
x0_axis, x1_axis = split_to_axis(sim.data[probe_a])
plt.plot(x0_axis, x1_axis)
plt.title(f'Radius of {r1}')
plt.xlabel('x0')
plt.ylabel('x1')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.axline((0, -1.5), (0, 1.5), color='black')
plt.axline((-1.5, 0), (1.5, 0), color='black')
plt.subplot(1, 2, 2)
plt.grid(True)
x0_axis, x1_axis = split_to_axis(sim.data[probe_b])
plt.plot(x0_axis, x1_axis)
plt.title(f'Radius of {r2}')
plt.xlabel('x0')
plt.ylabel('x1')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.axline((0, -1.5), (0, 1.5), color='black')
plt.axline((-1.5, 0), (1.5, 0), color='black')
plt.show()
