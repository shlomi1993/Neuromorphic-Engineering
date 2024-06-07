import nengo

import numpy as np
import matplotlib.pyplot as plt


# Transforming sin(x) to 2sin(x) by decoder scaling

T = 1.0
max_freq = 5

model = nengo.Network()
with model:
    stim = nengo.Node(lambda t: 0.5 * np.sin(10 * t))
    ens_a = nengo.Ensemble(n_neurons=100, dimensions=1)
    ens_b = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    nengo.Connection(stim, ens_a)
    nengo.Connection(ens_a, ens_b, transform=2)  # function=lambda x: 2*x)
    
    stim_p = nengo.Probe(stim)
    ens_a_p = nengo.Probe(ens_a, synapse=0.01)
    ens_b_p = nengo.Probe(ens_b, synapse=0.01)
    ens_a_spikes_p = nengo.Probe(ens_a.neurons, 'output')
    ens_b_spikes_p = nengo.Probe(ens_b.neurons, 'output')

sim = nengo.Simulator(model, seed=4)
sim.run(T)

t = sim.trange()
plt.figure(figsize=(6, 4))
plt.ax = plt.gca()
plt.plot(t, sim.data[stim_p], 'r', linewidth=4, label='x')
plt.plot(t, sim.data[ens_a_p], 'g', label='$\hat{x}$')
plt.plot(t, sim.data[ens_b_p], 'b', label='$f(\hat{x})=2\hat{x}$')
plt.legend()
plt.ylabel('Output')
plt.xlabel('Time')
plt.show()


# Transforming sin(x) to sin(x)^2

T = 1.0
max_freq = 5

model = nengo.Network()

with model:
    stim = nengo.Node(lambda t: 0.5 * np.sin(10 * t))
    ens_a = nengo.Ensemble(n_neurons=100, dimensions=1)
    ens_b = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    nengo.Connection(stim, ens_a)
    nengo.Connection(ens_a, ens_b, function=lambda x: x ** 2)
    
    stim_p = nengo.Probe(stim)
    ens_a_p = nengo.Probe(ens_a, synapse=0.01)
    ens_b_p = nengo.Probe(ens_b, synapse=0.01)
    ens_a_spikes_p = nengo.Probe(ens_a.neurons, 'output')
    ens_b_spikes_p = nengo.Probe(ens_b.neurons, 'output')

sim = nengo.Simulator(model, seed=4)
sim.run(T)

t = sim.trange()
plt.figure(figsize=(6, 4))
plt.ax = plt.gca()
plt.plot(t, sim.data[stim_p], 'r', linewidth=4, label='x')
plt.plot(t, sim.data[ens_a_p], 'g', label='$\hat{x}$')
plt.plot(t, sim.data[ens_b_p], 'b', label='$f(\hat{x})=\hat{x}^2$')
plt.legend()
plt.ylabel('Output')
plt.xlabel('Time')
plt.show()


# Summation of two shifted sin functions

T = 1.0
max_freq = 5

model = nengo.Network()

with model:
    stim_a = nengo.Node(lambda t: 0.5 * np.sin(10 * t))
    stim_b = nengo.Node(lambda t: 0.5 * np.sin(5 * t))
    
    ens_a = nengo.Ensemble(n_neurons=100, dimensions=1)
    ens_b = nengo.Ensemble(n_neurons=100, dimensions=1)
    ens_c = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    nengo.Connection(stim_a, ens_a)
    nengo.Connection(stim_b, ens_b)
    nengo.Connection(ens_a, ens_c)
    nengo.Connection(ens_b, ens_c)
    
    stim_a_p = nengo.Probe(stim_a)
    stim_b_p = nengo.Probe(stim_b)
    ens_a_p = nengo.Probe(ens_a, synapse=0.01)
    ens_b_p = nengo.Probe(ens_b, synapse=0.01)
    ens_c_p = nengo.Probe(ens_c, synapse=0.01)

sim = nengo.Simulator(model)
sim.run(T)

t = sim.trange()
plt.figure(figsize=(6, 4))
plt.plot(t, sim.data[ens_a_p], 'b', label='$\hat{x}$')
plt.plot(t, sim.data[ens_b_p], 'm--', label='$\hat{y}$')
plt.plot(t, sim.data[ens_c_p], 'k--', label='$\hat{x}+\hat{y}$')
plt.legend(loc='best')
plt.ylabel('Output')
plt.xlabel('Time')
plt.show()


# Summation of two encoded vectors

T = 1
max_freq = 5

model = nengo.Network()

with model:
    stim_a = nengo.Node([0.3, 0.5])
    stim_b = nengo.Node([0.3, -0.5])
    
    ens_a = nengo.Ensemble(n_neurons=100, dimensions=2)
    ens_b = nengo.Ensemble(n_neurons=100, dimensions=2)
    ens_c = nengo.Ensemble(n_neurons=100, dimensions=2)
    
    nengo.Connection(stim_a, ens_a)
    nengo.Connection(stim_b, ens_b)
    nengo.Connection(ens_a, ens_c)
    nengo.Connection(ens_b, ens_c)
    
    stim_a_p = nengo.Probe(stim_a)
    stim_b_p = nengo.Probe(stim_b)
    ens_a_p = nengo.Probe(ens_a, synapse=0.02)
    ens_b_p = nengo.Probe(ens_b, synapse=0.02)
    ens_c_p = nengo.Probe(ens_c, synapse=0.02)

sim = nengo.Simulator(model)
sim.run(T)

plt.figure()
plt.plot(sim.data[ens_a_p][:, 0], sim.data[ens_a_p][:, 1], 'g', label='$\hat{x}$')
plt.plot(sim.data[ens_b_p][:, 0], sim.data[ens_b_p][:, 1], 'm', label='$\hat{y}$')
plt.plot(sim.data[ens_c_p][:, 0], sim.data[ens_c_p][:, 1], 'k', label='$\hat{x} + \hat{y}$')
plt.ylabel('$x_2$')
plt.xlabel('$x_1$')
plt.legend(loc='best')
plt.show()


# Multiplication of two 2D vectors

T = 1.0
max_freq = 5

model = nengo.Network()

with model:
    stim_a = nengo.Node(lambda t: 0.5 * np.sin(10 * t))
    stim_b = nengo.Node(lambda t: 0.5 * np.sin(5 * t))
    
    ens_a = nengo.Ensemble(n_neurons=200, dimensions=2)
    ens_b = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    nengo.Connection(stim_a, ens_a[0])
    nengo.Connection(stim_b, ens_a[1])
    nengo.Connection(ens_a, ens_b, function=lambda x: x[0] * x[1])
    
    stim_a_p = nengo.Probe(stim_a)
    stim_b_p = nengo.Probe(stim_b)
    ens_a_p = nengo.Probe(ens_a, synapse=0.01)
    ens_b_p = nengo.Probe(ens_b, synapse=0.01)

sim = nengo.Simulator(model)
sim.run(T)

t = sim.trange()
plt.figure()
plt.plot(t, sim.data[ens_a_p][:, 0], 'black', label='$\hat{x}[0]$')
plt.plot(t, sim.data[ens_a_p][:, 1], 'black', label='$\hat{x}[1]$')
plt.plot(t, sim.data[ens_b_p], 'r', label='$\hat{x[0]}\cdot\hat{x[1]}$')
plt.legend(loc='best')
plt.ylabel('Output')
plt.xlabel('Time')
plt.show()


# Signal gating

T = 1.0
max_freq = 5

model = nengo.Network()

with model:
    stim_a = nengo.Node(lambda t: 0.5 * np.sin(10 * t))
    stim_b = nengo.Node(lambda t: 0 if t < 0.5 else 1)
    
    ens_a = nengo.Ensemble(n_neurons=300, dimensions=2, radius=np.sqrt(2))
    ens_b = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    nengo.Connection(stim_a, ens_a[0])
    nengo.Connection(stim_b, ens_a[1])
    nengo.Connection(ens_a, ens_b, function=lambda x: x[0] * x[1])
    
    stim_a_p = nengo.Probe(stim_a)
    stim_b_p = nengo.Probe(stim_b)
    ens_a_p = nengo.Probe(ens_a, synapse=0.01)
    ens_b_p = nengo.Probe(ens_b, synapse=.01)
   
sim = nengo.Simulator(model)
sim.run(T)

t = sim.trange()
plt.figure()
plt.plot(t, sim.data[ens_a_p][:,0],'black', label='$\hat{x}[0]$')
plt.plot(t, sim.data[ens_a_p][:,1],'blue', label='$\hat{x}[1]$')
plt.plot(t, sim.data[ens_b_p],'r', label='$\hat{x[0]}\cdot\hat{x[1]}$')
plt.legend(loc='best')
plt.ylabel('Output')
plt.xlabel('Time')
plt.show()