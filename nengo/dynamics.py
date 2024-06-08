import nengo
import matplotlib.pyplot as plt

from nengo.processes import Piecewise


# Recurrent computing of f(x) = x + 1

model = nengo.Network()
with model:
    ens_a = nengo.Ensemble(n_neurons=100, dimensions=1)
    ens_b = nengo.Ensemble(n_neurons=100, dimensions=1)
    ens_c = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    def feedback(x):
        return x + 1
    
    nengo.Connection(ens_a, ens_a, function=feedback, synapse=0.1)
    nengo.Connection(ens_b, ens_b, function=feedback, synapse=0.2)
    nengo.Connection(ens_c, ens_c, function=feedback, synapse=0.3)

    ens_a_p = nengo.Probe(ens_a, synapse=0.01)
    ens_b_p = nengo.Probe(ens_b, synapse=0.01)
    ens_c_p = nengo.Probe(ens_c, synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(0.5)

t = sim.trange()
plt.plot(t, sim.data[ens_a_p], label='$tau=0.1$')
plt.plot(t, sim.data[ens_b_p], label='$tau=0.2$')
plt.plot(t, sim.data[ens_c_p], label='$tau=0.3$')
plt.legend()
plt.ylabel('Output')
plt.xlabel('Time')
plt.ylim(0, 1.5)
plt.show()


# Recurrent computing of f(x) = -x

with model:
    stim = nengo.Node(Piecewise({0: 1, 0.2: -1, 0.4: 0}))
    
    def feedback(x):
        return -x
    
    ens_a = nengo.Ensemble(n_neurons=100, dimensions=1)
    ens_b = nengo.Ensemble(n_neurons=100, dimensions=1)
    ens_c = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    nengo.Connection(stim, ens_a)
    nengo.Connection(stim, ens_b)
    nengo.Connection(stim, ens_c)
    
    nengo.Connection(ens_a, ens_a, function=feedback, synapse=0.1)
    nengo.Connection(ens_b, ens_b, function=feedback, synapse=0.2)
    nengo.Connection(ens_c, ens_c, function=feedback, synapse=0.3)
    
    ens_a_p = nengo.Probe(ens_a, synapse=0.01)
    ens_b_p = nengo.Probe(ens_b, synapse=0.01)
    ens_c_p = nengo.Probe(ens_c, synapse=0.01)
    stim_p = nengo.Probe(stim, synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(0.6)

t = sim.trange()
plt.plot(t, sim.data[stim_p], 'r', label='stimulus')
plt.plot(t, sim.data[ens_a_p], label='$tau=0.01$')
plt.plot(t, sim.data[ens_b_p], label='$tau=0.02$')
plt.plot(t, sim.data[ens_c_p], label='$tau=0.03$')
plt.legend()
plt.ylabel('Output')
plt.xlabel('Time')
plt.show()


# Recurrent computing of f(x) = x^2

with model:
    stim = nengo.Node(Piecewise({0.1: 0.2, 0.2: 0.4, 0.6: 0}))
    
    def feedback(x):
        return x * x

    ens_a = nengo.Ensemble(n_neurons=100, dimensions=1)
    ens_b = nengo.Ensemble(n_neurons=100, dimensions=1)
    ens_c = nengo.Ensemble(n_neurons=100, dimensions=1)

    nengo.Connection(stim, ens_a)
    nengo.Connection(stim, ens_b)
    nengo.Connection(stim, ens_c)

    nengo.Connection(ens_a, ens_a, function=feedback, synapse=0.1)
    nengo.Connection(ens_b, ens_b, function=feedback, synapse=0.2)
    nengo.Connection(ens_c, ens_c, function=feedback, synapse=0.3)

    ens_a_p = nengo.Probe(ens_a, synapse=0.01)
    ens_b_p = nengo.Probe(ens_b, synapse=0.01)
    ens_c_p = nengo.Probe(ens_c, synapse=0.01)
    stim_p = nengo.Probe(stim, synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(1)

t = sim.trange()
plt.plot(t, sim.data[stim_p], 'r', label='stimulus')
plt.plot(t, sim.data[ens_a_p], label='$tau=0.01$')
plt.plot(t, sim.data[ens_b_p], label='$tau=0.02$')
plt.plot(t, sim.data[ens_c_p], label='$tau=0.03$')
plt.legend()
plt.ylabel('Output')
plt.xlabel('Time')
plt.show()


# Integrator

tau1 = 0.001
tau2 = 0.01
tau3 = 1
model = nengo.Network('Eye control', seed=8)
with model:
    stim = nengo.Node(Piecewise({0.3: 1, 0.6: 0}))
    velocity = nengo.Ensemble(100, dimensions=1)
    position1 = nengo.Ensemble(200, dimensions=1)
    position2 = nengo.Ensemble(200, dimensions=1)
    position3 = nengo.Ensemble(200, dimensions=1)

    def feedback(x):
        return x

    nengo.Connection(stim, velocity)
    nengo.Connection(velocity, position1, transform=tau1, synapse=tau1)
    nengo.Connection(position1, position1, function=feedback, synapse=tau1)
    nengo.Connection(velocity, position2, transform=tau2, synapse=tau2)
    nengo.Connection(position2, position2, function=feedback, synapse=tau2)
    nengo.Connection(velocity, position3, transform=tau3, synapse=tau3)
    nengo.Connection(position2, position3, function=feedback, synapse=tau3)

    stim_p = nengo.Probe(stim)
    velocity_p = nengo.Probe(velocity, synapse=0.01)
    position_p = nengo.Probe(position1, synapse=0.01)
    position_p2 = nengo.Probe(position2, synapse=0.01)
    position_p3 = nengo.Probe(position3, synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(1)

plt.figure()
plt.plot(sim.trange(), sim.data[stim_p], label='Input', linewidth=4, color='black')
plt.plot(sim.trange(), sim.data[velocity_p], label='velocity', linewidth=2)
plt.plot(sim.trange(), sim.data[position_p], label='Position (tau=0.001)', linewidth=2)
plt.plot(sim.trange(), sim.data[position_p2], label='position (tau=0.1)', linewidth=2)
plt.plot(sim.trange(), sim.data[position_p3], label='position (tau=1)', linewidth=2)
plt.ylabel('Output')
plt.xlabel('Time')
plt.legend(loc='best')
plt.show()


# Leaky integrator

tau = 0.1
tau_c = 2.0

model = nengo.Network('Eye control', seed=5)
with model:
    stim = nengo.Node(Piecewise({0.3: 1, 0.6: 0}))
    velocity = nengo.Ensemble(n_neurons=100, dimensions=1)
    position = nengo.Ensemble(n_neurons=200, dimensions=1)
    
    def feedback(x):
        return (-tau / tau_c + 1) * x

    nengo.Connection(stim, velocity)
    nengo.Connection(velocity, position, transform=tau, synapse=tau)
    nengo.Connection(position, position, function=feedback, synapse=tau)

    stim_p = nengo.Probe(stim)
    position_p = nengo.Probe(position, synapse=0.01)
    velocity_p = nengo.Probe(velocity, synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(3)

t = sim.trange()
plt.plot(t, sim.data[stim_p], label='stim')
plt.plot(t, sim.data[position_p], label='position')
plt.plot(t, sim.data[velocity_p], label='velocity')
plt.ylabel('Output')
plt.xlabel('Time')
plt.legend(loc='best')
plt.show()


# Controlled leaky integrator

tau = 0.1

model = nengo.Network('Controlled integrator', seed=1)
with model:
    vel = nengo.Node(Piecewise({.2:1.5, .5:0}))
    dec = nengo.Node(Piecewise({.7:.2, .9:0}))
    
    velocity = nengo.Ensemble(n_neurons=100, dimensions=1)
    decay = nengo.Ensemble(n_neurons=100, dimensions=1)
    position = nengo.Ensemble(n_neurons=400, dimensions=2)
    
    def feedback(x):
        return -x[1] * x[0] + x[0], 0
    
    nengo.Connection(vel, velocity)
    nengo.Connection(dec, decay)

    nengo.Connection(velocity, position[0], transform=tau, synapse=tau)
    nengo.Connection(decay, position[1], synapse=0.01)
    nengo.Connection(position, position, function=feedback, synapse=tau)

    position_p = nengo.Probe(position[0], synapse=.01)
    velocity_p = nengo.Probe(velocity, synapse=.01)
    decay_p = nengo.Probe(decay, synapse=.01)

with nengo.Simulator(model) as sim:
    sim.run(1)

plt.plot(sim.trange(), sim.data[decay_p])
plt.lineObjects = plt.plot(sim.trange(), sim.data[position_p])
plt.plot(sim.trange(), sim.data[velocity_p])
plt.ylabel('Output')
plt.xlabel('Time')
plt.legend(('decay','position', 'velocity'), loc='best')
plt.show()


# Oscillator

freq = -0.25
model = nengo.Network('Oscillator')
with model:
    stim = nengo.Node(lambda t: [0.5, 0.5] if t < 0.02 else [0, 0])
    osc = nengo.Ensemble(n_neurons=200, dimensions=2)
    
    def feedback(x):
        return x[0] + freq * x[1], -freq * x[0] + x[1]
    
    nengo.Connection(osc, osc, function=feedback, synapse=.01)
    nengo.Connection(stim, osc)
    
    stim_p = nengo.Probe(stim)
    osc_p = nengo.Probe(osc, synapse=0.01)
    
with nengo.Simulator(model) as sim:
    sim.run(.5)

t = sim.trange()
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(t, sim.data[osc_p])
plt.plot(t, sim.data[stim_p], 'r', label = 'stim', linewidth=4)
plt.xlabel('Time (s)')
plt.ylabel('State value')       
plt.subplot(1, 2, 2)
plt.plot(sim.data[osc_p][:,0],sim.data[osc_p][:,1])
plt.xlabel('$x_0$')
plt.ylabel('$x_1$')
plt.show()


# Controlled oscillator

freq = -0.25
model = nengo.Network('Oscillator')
with model:
    stim = nengo.Node(lambda t: [0.5, 0.5] if t < 0.02 else [0,0])
    freq_ctrl = nengo.Node(Piecewise({0:-0.1, 4:-.2, 8:-0.3}))
    osc = nengo.Ensemble(n_neurons=200, dimensions=3)

    def feedback(x):
        return x[0] + x[2] * x[1], -x[2] * x[0] + x[1], 0

    nengo.Connection(osc, osc, function=feedback, synapse=0.01)
    nengo.Connection(stim, osc[0:2])
    nengo.Connection(freq_ctrl, osc[2])
    
    stim_p = nengo.Probe(stim)
    osc_p = nengo.Probe(osc, synapse=.01)

with nengo.Simulator(model) as sim:
    sim.run(12)

t = sim.trange()
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(t, sim.data[osc_p])
plt.plot(t, sim.data[stim_p], 'r', label = 'stim', linewidth=4)
plt.xlabel('Time (s)')
plt.ylabel('State value')       
plt.subplot(1,2,2)
plt.plot(sim.data[osc_p][:,0],sim.data[osc_p][:,1])
plt.xlabel('$x_0$')
plt.ylabel('$x_1$')
plt.show()


# Point attractor

freq = -0.25
model = nengo.Network(label='Oscillator')
with model:
    stim = nengo.Node(lambda t: [.5, .5] if t < 0.01 else [0, 0])
    osc = nengo.Ensemble(n_neurons=2000, dimensions=2)
    
    def feedback(x):
        p1 = 0.5
        p2 = 0.8
        return [x[0] - (x[0] - p1), x[1] - (x[1] - p2)]
    
    nengo.Connection(osc, osc, function=feedback, synapse=0.01)
    nengo.Connection(stim, osc)
    
    stim_p = nengo.Probe(stim)
    osc_p = nengo.Probe(osc, synapse=0.01)
    
with nengo.Simulator(model) as sim:
    sim.run(0.5)

t = sim.trange()
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(t, sim.data[osc_p])
plt.plot(t, sim.data[stim_p], 'r', label='stim', linewidth=4)
plt.xlabel('Time (s)')
plt.ylabel('State value')
plt.subplot(1, 2, 2)
plt.plot(sim.data[osc_p][:, 0], sim.data[osc_p][:, 1])
plt.xlabel('$x_0$')
plt.ylabel('$x_1$')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()


# 2D point attractor

N = 200
tau = 0.01
model = nengo.Network(label='2D Plane Attractor', seed=4)
with model:
    stim = nengo.Node(Piecewise({0.3: [1, 0], 0.5: [0, 0], 0.7: [0, -1], 0.9: [0, 0]}))
    neurons1 = nengo.Ensemble(n_neurons=N, dimensions=2)
    neurons2 = nengo.Ensemble(n_neurons=N * 10, dimensions=2)

    nengo.Connection(stim, neurons1, transform=tau, synapse=tau)
    nengo.Connection(neurons1, neurons1, synapse=tau)
    nengo.Connection(stim, neurons2, transform=tau, synapse=tau)
    nengo.Connection(neurons2, neurons2, synapse=tau)

    stim_p = nengo.Probe(stim)
    neurons_p1 = nengo.Probe(neurons1, synapse=0.01)
    neurons_p2 = nengo.Probe(neurons2, synapse=0.01)
    
sim = nengo.Simulator(model)
sim.run(4)

t = sim.trange()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(t, sim.data[stim_p], label='stim')
plt.plot(t, sim.data[neurons_p1], label='position')
plt.ylabel('Output')
plt.xlabel('Time')
plt.legend(loc='best')
plt.title('# neurons = 200')

plt.subplot(1, 2, 2)
plt.plot(t, sim.data[stim_p], label='stim')
plt.plot(t, sim.data[neurons_p2], label='position')
plt.ylabel('Output')
plt.xlabel('Time')
plt.legend(loc='best')
plt.title('# neurons = 2000')
plt.show()


# 2D Plane Attractor - Ensemble Array

N = 500  # neurons per sub_ensemble
tau = 0.01
model = nengo.Network()
with model:
    stim = nengo.Node(Piecewise({0.5: [1, 0], 1: [0, 0], 2: [0, -1], 2.5: [0, 0]}))
    neurons = nengo.networks.EnsembleArray(n_neurons=N, n_ensembles=2, seed=6)
    
    nengo.Connection(stim, neurons.input, transform=tau, synapse=tau)
    nengo.Connection(neurons.output, neurons.input, synapse=tau)

    stim_p = nengo.Probe(stim)
    neurons_p = nengo.Probe(neurons.output, synapse=0.01)
    
with nengo.Simulator(model) as sim:
    sim.run(4)

t = sim.trange()
plt.plot(t, sim.data[stim_p], label='stim')
plt.plot(t, sim.data[neurons_p], label='position')
plt.ylabel('Output')
plt.xlabel('Time')
plt.legend(loc='best')
plt.show()


# Lorenz Attractor

model = nengo.Network(label='Lorenz Attractor', seed=3)
with model:
    x = nengo.Ensemble(n_neurons=600, dimensions=3, radius=30)
    synapse = 0.1
    
    def lorenz(x):
        sigma = 10
        beta = 8.0 / 3
        rho = 28
        dx0 = -sigma * x[0] + sigma * x[1]
        dx1 = -x[0] * x[2] - x[1]
        dx2 = x[0] * x[1] - beta * (x[2] + rho) - rho
        return [dx0 * synapse + x[0],
                dx1 * synapse + x[1],
                dx2 * synapse + x[2]]
    
    nengo.Connection(x, x, synapse=synapse, function=lorenz)
    
    lorenz_p = nengo.Probe(x, synapse=0.01)
    
with nengo.Simulator(model) as sim:
    sim.run(14)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(sim.trange(), sim.data[lorenz_p][:, 0], label='$x_0$')
plt.plot(sim.trange(), sim.data[lorenz_p][:, 1], label='$x_1$')
plt.plot(sim.trange(), sim.data[lorenz_p][:, 2], label='$x_2$')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('State value')
plt.subplot(1, 2, 2)
plt.plot(sim.data[lorenz_p][:, 0], sim.data[lorenz_p][:, 1])
plt.xlabel('$x_0$')
plt.ylabel('$x_1$')
plt.show()
