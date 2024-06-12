import numpy as np
import matplotlib.pyplot as plt
import nengo

from nengo.utils.matplotlib import rasterplot
from nengo.dists import Uniform
from nengo.utils.ensemble import tuning_curves


plt.rc('font', size=14, weight='bold')


# Rectified linear and NEF LIF neurons
J = np.linspace(-1, 10, 100)
plt.figure(figsize=(8, 6))
plt.plot(J, nengo.neurons.LIFRate(tau_rc=0.02, tau_ref=0.002).rates(J, gain=1, bias=0)) 
plt.xlabel('I')
plt.ylabel('a (Hz)'); 
plt.show()


# Model definition
model = nengo.Network(label='Tuning Curves')
with model:
    stim = nengo.Node(lambda t: np.sin(10 * t))
    ens1 = nengo.Ensemble(n_neurons=2, dimensions=1, encoders=[[1],[-1]], intercepts=[-.5, -.5], max_rates=[100, 100])
    ens2 = nengo.Ensemble(n_neurons=50, dimensions=1, max_rates=Uniform(100, 200))
    nengo.Connection(stim, ens1)
    nengo.Connection(stim, ens2)
    
    probe_stim = nengo.Probe(stim)
    probe1 = nengo.Probe(ens1.neurons, 'output')
    probe2 = nengo.Probe(ens2.neurons, 'output')


# Model simulation
with nengo.Simulator(model) as sim:
    sim.run(0.6)


# Plot results
t = sim.trange()
for ens, probe in [(ens1, probe1), (ens2, probe2)]:
    fig = plt.figure(figsize=(24, 6))

    plt.subplot(1, 3, 1)
    plt.title(f'{ens.n_neurons} Neuron Tuning Curves')
    plt.plot(*tuning_curves(ens, sim))
    plt.xlabel('I')
    plt.ylabel('a (Hz)')

    plt.subplot(1, 3, 2)
    plt.title(f'Encoding by {ens.n_neurons} Neurons')
    plt.plot(t, sim.data[probe_stim],'r', linewidth=4)
    plt.ax = plt.gca()
    plt.xlabel('Time')
    plt.ylabel('Encoded value')
    rasterplot(t, sim.data[probe], ax=plt.ax.twinx(), use_eventplot=True)

    x = sim.data[probe_stim][:,0]
    A = sim.data[probe]
    gamma = np.dot(A.T,A)
    upsilon = np.dot(A.T,x)
    d = np.dot(np.linalg.pinv(gamma), upsilon)
    xhat = np.dot(A, d)

    plt.subplot(1, 3, 3)
    plt.title(f'Decoding by {ens.n_neurons} Neurons')
    plt.plot(t, x, label='Stimulus', color='r', linewidth=4)
    plt.plot(t, xhat, label='Decoded stimulus')
    plt.xlabel('Time')
    plt.ylabel('Decoded value')

    fig.tight_layout(pad=5.0)
    plt.show()
