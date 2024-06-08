import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import seaborn
import nengo

from nengo.utils.matplotlib import rasterplot
from nengo.dists import Uniform
from nengo.dists import Choice
from nengo.utils.ensemble import tuning_curves


# Rectified linear and NEF LIF neurons

n_RL  = nengo.neurons.RectifiedLinear()
n_LIF = nengo.neurons.LIFRate(tau_rc=0.02, tau_ref=0.002)

J = np.linspace(-1, 10, 100)

plt.plot(J, n_RL.rates(J, gain=10, bias=0))
plt.xlabel('I')
plt.ylabel('$a$ (Hz)')
plt.show()
plt.plot(J, n_LIF.rates(J, gain=1, bias=0)) 
plt.xlabel('I')
plt.ylabel('a (Hz)'); 
plt.show()


#  LIF neuron response to a sinusoidal input

model = nengo.Network(label='One Neuron')
with model:
    stimulus = nengo.Node(lambda t: np.sin(10 * t))
    ens = nengo.Ensemble(n_neurons=1, dimensions=1, encoders = [[1]], intercepts = [0.5], max_rates= [100])
    nengo.Connection(stimulus, ens)
    spikes_p = nengo.Probe(ens.neurons, 'output')
    voltage_p = nengo.Probe(ens.neurons, 'voltage')
    stim_p = nengo.Probe(stimulus)

with nengo.Simulator(model) as sim:
    sim.run(1)

t = sim.trange()     
plt.figure(figsize=(10, 4))
plt.plot(t, sim.data[stim_p], label='stimulus', color='r', linewidth=4)
plt.ax = plt.gca()
plt.ax.plot(t, sim.data[voltage_p],'g', label='v')
plt.ylim((-1, 2))
plt.ylabel('Voltage')
plt.xlabel('Time')
plt.legend(loc='lower left')
rasterplot(t, sim.data[spikes_p], ax=plt.ax.twinx(), use_eventplot=True)
plt.ylim((-1, 2))
plt.show()


# Two LIF neurons with the same intercept (0.5) and  op-posing encoders

model = nengo.Network(label='Two Neurons')
with model:
    stim = nengo.Node(lambda t: np.sin(10 * t))
    ens = nengo.Ensemble(n_neurons=2, dimensions=1, encoders=[[1],[-1]], intercepts=[-.5, -.5], max_rates=[100, 100])
    nengo.Connection(stim, ens)
    stim_p = nengo.Probe(stim)
    spikes_p = nengo.Probe(ens.neurons, 'output')

with nengo.Simulator(model) as sim:
    sim.run(.6)

t = sim.trange()
plt.plot(*tuning_curves(ens, sim))
plt.xlabel('I')
plt.ylabel('$a$ (Hz)')
plt.show()
plt.figure(figsize=(12, 6))
plt.plot(t, sim.data[stim_p],'r', linewidth=4)
plt.ax = plt.gca()
plt.ylabel('Output')
plt.xlabel('Time')
rasterplot(t, sim.data[spikes_p], ax=plt.ax.twinx(), use_eventplot=True)
plt.ylabel('Neuron')
plt.show()


# 50 LIF neurons with uniformly distributed maximal spiking rates and randomized intercepts.

model = nengo.Network(label='Decoding Neurons')
with model:
    stim = nengo.Node(lambda t: np.sin(10 * t))
    ens = nengo.Ensemble(n_neurons=50, dimensions=1, max_rates=Uniform(100, 200))
    nengo.Connection(stim, ens)   
    stim_p = nengo.Probe(stim)
    spikes_p = nengo.Probe(ens.neurons, 'output')

with nengo.Simulator(model) as sim:
    sim.run(.6)

x = sim.data[stim_p][:,0]
A = sim.data[spikes_p]

t = sim.trange()
plt.plot(*tuning_curves(ens, sim))
plt.xlabel('I')
plt.ylabel('$a$ (Hz)')
plt.show()
plt.figure(figsize=(12, 6))
plt.ax = plt.gca()
plt.plot(t, sim.data[stim_p],'r', linewidth=4)
plt.ylabel('Output')
plt.xlabel('Time')
rasterplot(t, sim.data[spikes_p], ax=plt.ax.twinx(), use_eventplot=True, color='k')
plt.ylabel('Neuron')
plt.show()


# Two LIF-based stimulus decoding

model = nengo.Network(label='Decoding Neurons')
with model:
    stim = nengo.Node(lambda t: np.sin(10 * t))
    ens = nengo.Ensemble(n_neurons=2, dimensions=1, encoders=[[1],[-1]], intercepts=[-.5, -.5], max_rates=[100, 100])
    nengo.Connection(stim, ens)
    stim_p = nengo.Probe(stim)
    spikes_p = nengo.Probe(ens.neurons, 'output')

with nengo.Simulator(model) as sim:
    sim.run(.6)

t = sim.trange()
x = sim.data[stim_p][:,0]
A = sim.data[spikes_p]

gamma = np.dot(A.T,A)
upsilon = np.dot(A.T,x)
d = np.dot(np.linalg.pinv(gamma), upsilon)

xhat = np.dot(A, d)

plt.figure(figsize=(8,4))
plt.plot(t, x, label='Stimulus', color='r', linewidth=4)
plt.plot(t, xhat, label='Decoded stimulus')
plt.ylabel('$x$')
plt.xlabel('Time')
plt.show()


# 50 LIF-based stimulus decoding

model = nengo.Network(label='Decoding Neurons')
with model:
    stim = nengo.Node(lambda t: np.sin(10 * t))
    ens = nengo.Ensemble(n_neurons=50, dimensions=1,max_rates=Uniform(100,200))
    temp = nengo.Ensemble(10, dimensions=1)
    nengo.Connection(stim, ens)
    connection = nengo.Connection(ens, temp)  # This is just to generate the decoders
    stim_p = nengo.Probe(stim)
    spikes_p = nengo.Probe(ens.neurons, 'output')

with nengo.Simulator(model) as sim:
    sim.run(.6)

x = sim.data[stim_p][:,0]

A = sim.data[spikes_p]

gamma = np.dot(A.T, A)
upsilon = np.dot(A.T, x)
d = np.dot(np.linalg.pinv(gamma), upsilon)

xhat = np.dot(A, d)

t = sim.trange()
plt.figure(figsize=(12, 6))
plt.ax = plt.gca()
plt.plot(t, sim.data[stim_p], 'r', linewidth=4)
plt.plot(t, xhat)
plt.ylabel('x')
plt.xlabel('Time')
plt.show()


# Exponentially decaying filters

dt = 0.001

tau_gaba = 0.010
tau_ampa = 0.002
tau_nmda = 0.145

t_h = np.arange(1000) * dt - 0.5
h_gaba = np.exp(-t_h / tau_gaba)
h_ampa = np.exp(-t_h / tau_ampa)
h_nmda = np.exp(-t_h / tau_nmda)

h_gaba[np.where(t_h < 0)] = 0
h_gaba = h_gaba / np.linalg.norm(h_gaba, 1)

h_ampa[np.where(t_h < 0)]=0
h_ampa = h_ampa / np.linalg.norm(h_ampa, 1)

h_nmda[np.where(t_h < 0)] = 0
h_nmda = h_nmda / np.linalg.norm(h_nmda, 1)

plt.figure()
plt.plot(t_h, h_gaba, label='GABA', linewidth=4)
plt.plot(t_h, h_ampa, label='AMPA', linewidth=4)
plt.plot(t_h, h_nmda, label='NMDA', linewidth=4)

plt.legend()
plt.xlabel('t')
plt.ylabel('h(t)')
plt.xlim((-0.01, 0.04))
plt.show()


# Two convolved LIF-based stimulus decodings

dt = 0.001
tau = 0.05
t_h = np.arange(1000) * dt - 0.5
h = np.exp(-t_h / tau)
h[np.where(t_h < 0)] = 0
h = h / np.linalg.norm(h, 1)

model = nengo.Network(label='Decoding Neurons')
with model:
    stim = nengo.Node(lambda t: np.sin(10 * t))
    ens = nengo.Ensemble(n_neurons=2, dimensions=1, encoders = [[1],[-1]], intercepts=[-.5, -.5], max_rates= [100, 100])
    nengo.Connection(stim, ens)
    temp = nengo.Ensemble(n_neurons=1, dimensions=1)
    connection = nengo.Connection(ens, temp)  # Generating the decoders
    stim_p = nengo.Probe(stim)
    spikes_p = nengo.Probe(ens.neurons, 'output')

with nengo.Simulator(model) as sim:
    sim.run(1)

sig = sim.data[stim_p][:,0]

fspikes1 = np.convolve(sim.data[spikes_p][:,0], h, mode='same')
fspikes2 = np.convolve(sim.data[spikes_p][:,1], h, mode='same')

A = np.array([fspikes1, fspikes2]).T
d = sim.data[connection].weights.T  
xhat = np.dot(A, d)
t = sim.trange()

plt.figure(figsize=(8,4))
plt.ax = plt.gca()
plt.plot(t, sig,'r',linewidth=4)
plt.plot(t, fspikes1*d[0],linewidth=4)
plt.plot(t, fspikes2*d[1],linewidth=4)
plt.ylabel('$x$')
rasterplot(t, sim.data[spikes_p], ax=plt.ax.twinx(), use_eventplot=True, color='k')
plt.xlabel('Time (s)')

plt.figure(figsize=(8,4))
plt.plot(t, sig, label='x',color='r',linewidth=4)
plt.plot(t, xhat, label='x')
plt.ylabel('$x$')
plt.ylim(-1,1)
plt.xlabel('Time (s)')
plt.show()


# 50 convolved LIF-based stimulus decoding.

model = nengo.Network(label='Decoding Neurons')
with model:
    stim = nengo.Node(lambda t: np.sin(10 * t))  
    ens = nengo.Ensemble(n_neurons=50, dimensions=1, max_rates=Uniform(100, 200))
    nengo.Connection(stim, ens)
    stim_p = nengo.Probe(stim)
    dec_p = nengo.Probe(ens, synapse=0.05)
    spikes_p = nengo.Probe(ens.neurons, 'output')
    
with nengo.Simulator(model) as sim:
    sim.run(1)

sig = sim.data[stim_p][:,0]

t = sim.trange()
plt.figure(figsize=(8,4))
plt.ax = plt.gca()
plt.plot(t,sim.data[dec_p], linewidth=4)
plt.plot(t,sig, c='r', linewidth=4)
plt.ylabel('$x$')
plt.xlabel('Time (s)');
rasterplot(t, sim.data[spikes_p], ax=plt.ax.twinx(), use_eventplot=True, color='k')
plt.show()


# Representation of f(x) = x (randomly distributed tuning)

model = nengo.Network(label='Neurons')
with model:
    neurons = nengo.Ensemble(n_neurons=50, dimensions=1) 
    connection = nengo.Connection(neurons, neurons)  # This is just to generate the decoders
    
sim = nengo.Simulator(model)

d = sim.data[connection].weights.T
x, A = tuning_curves(neurons, sim)
xhat = np.dot(A, 1 * d)

x = 1 * x
plt.figure(figsize=(3, 4))
plt.plot(x, A)
plt.xlabel('x')
plt.ylabel('firing rate (Hz)')
plt.show()

plt.figure()
plt.plot(x, x, linewidth=4, label='f(x)=x')
plt.plot(x, xhat, 'r', linewidth=4, label='$\hat{x}$')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend()
plt.show()


# Representation of f(x) = x (uniformly distributed tuning)

model = nengo.Network(label='Neurons')
with model:
    neurons = nengo.Ensemble(n_neurons=50, dimensions=1, intercepts=Uniform(-.3,.3)) 
    connection = nengo.Connection(neurons, neurons) #This is just to generate the decoders
    
sim = nengo.Simulator(model)

d = sim.data[connection].weights.T
x, A = tuning_curves(neurons, sim)
xhat = np.dot(A, 1*d)

x= 1*x
plt.figure(figsize=(3,4))
plt.plot(x, A)
plt.xlabel('x')
plt.ylabel('firing rate (Hz)')
plt.show()

plt.figure()
plt.plot(x, x, linewidth=4, label='f(x)=x')
plt.plot(x, xhat, 'r', linewidth=4, label='$\hat{x}$')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend()
plt.show()


# Representation of f(x) = x (intercepts = -0.2)

model = nengo.Network(label='Neurons')
with model:
    neurons = nengo.Ensemble(n_neurons=50, dimensions=1, intercepts = Choice([-0.2])) 
    connection = nengo.Connection(neurons, neurons) #This is just to generate the decoders
    
sim = nengo.Simulator(model)

d = sim.data[connection].weights.T
x, A = tuning_curves(neurons, sim)
xhat = np.dot(A, 1 * d)

x = 1 * x
plt.figure(figsize=(3,4))
plt.plot(x, A)
plt.xlabel('x')
plt.ylabel('firing rate (Hz)')
plt.show()

plt.figure()
plt.plot(x, x, linewidth=4, label='f(x)=x')
plt.plot(x, xhat, 'r', linewidth=4, label='$\hat{x}$')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend()
plt.show()


# Five most important basis functions and variation drop for 1,000 randomly tuned neurons.

model = nengo.Network(label='Neurons')
with model:
    neurons = nengo.Ensemble(n_neurons=1000, dimensions=1) 
    connection = nengo.Connection(neurons, neurons)  # This is just to generate the decoders
    
sim = nengo.Simulator(model)

d = sim.data[connection].weights.T
x, A = tuning_curves(neurons, sim)
xhat = np.dot(A, d)

Gamma = np.dot(A.T, A)
U, S, V = np.linalg.svd(Gamma)
chi = np.dot(A, U)

for i in range(5):
    plt.plot(x, chi[:,i], label='$\chi_%d$=%1.3g'%(i, S[i]), linewidth=3)
plt.legend(loc='best')
plt.figure()
plt.xlabel('neuron')
plt.loglog(S, linewidth=4)
plt.show()


# Five most important basis functions and variation drop for 1,000 uniformly tuned neurons.

model = nengo.Network(label='Neurons')
with model:
    neurons = nengo.Ensemble(n_neurons=1000, dimensions=1, intercepts=Uniform(-.3,.3)) 
    connection = nengo.Connection(neurons, neurons) #This is just to generate the decoders
    
sim = nengo.Simulator(model)

d = sim.data[connection].weights.T
x, A = tuning_curves(neurons, sim)
xhat = np.dot(A, d)

Gamma = np.dot(A.T, A)
U,S1,V = np.linalg.svd(Gamma)
chi = np.dot(A, U)

for i in range(5):
    plt.plot(x, chi[:,i], label='$\chi_%d$=%1.3g'%(i, S1[i]), linewidth=3)
plt.legend(loc='best')    
plt.figure()
plt.xlabel('neuron')
plt.loglog(S1, linewidth=4, label='intercepts=Uniform(-.3,.3)')
plt.legend()
plt.show()


# High dimensional representation

model = nengo.Network()
with model:
    ens_2d = nengo.Ensemble(n_neurons=4, dimensions=2, encoders=Choice([[1, 1]]), seed=1)
with nengo.Simulator(model) as sim:
    eval_points, activities = tuning_curves(ens_2d, sim)

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection='3d')
for i in range(ens_2d.n_neurons):
    ax.plot_surface(eval_points.T[0], eval_points.T[1], activities.T[i], alpha=0.5)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('Firing rate (Hz)')
plt.show()


# Four basis functions for a 500 2D neurons ensemble

model = nengo.Network(label='Neurons', seed=2)
with model:
    neurons = nengo.Ensemble(n_neurons=500, dimensions=2) 
    connection = nengo.Connection(neurons, neurons)  # This is just to generate the decoders
    
sim = nengo.Simulator(model)

d = sim.data[connection].weights.T
x, A = tuning_curves(neurons, sim)
A = np.reshape(A, (2500, 500))
               
Gamma = np.dot(A.T, A)
U, S, V = np.linalg.svd(Gamma)
chi = np.dot(A, U)

for index in [1, 3, 5, 7]:  #W hat's 0? 3, 4, 5 (same/diff signs, cross)? Higher?
    basis = chi[:,index]
    basis.shape = 50,50

    x0 = np.linspace(-1, 1, 50)
    x1 = np.linspace(-1, 1, 50)
    x0, x1 = np.array(np.meshgrid(x0,x1))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    p = ax.plot_surface(x0, x1, basis, linewidth=0, cstride=1, rstride=1)
    plt.locator_params(nbins=5)
    plt.show()



# Activity analysis of ensembles with various dimensions

def analytic_proportion(x, d):
    flip = False
    if x < 0: 
        x = -x
        flip = True
    value = 0 if x >= 1.0 else 0.5 * scipy.special.betainc((d + 1) / 2.0, 0.5, 1 - x ** 2)
    if flip:
        value = 1.0 - value
    return value

def plot_intercept_distribution(ens):
    
    pts = ens.eval_points.sample(n=1000, d=ens.dimensions)
    model = nengo.Network()
    model.ensembles.append(ens)
    sim = nengo.Simulator(model)
    _, activity = nengo.utils.ensemble.tuning_curves(ens, sim, inputs=pts)
    p = np.mean(activity>0, axis=0)
    seaborn.distplot(p)
    plt.xlabel('Activity proportion')

plt.rcParams.update({'font.size': 12})

for D in [1, 4, 32]:
    plt.figure(figsize=(6,4))
    ens = nengo.Ensemble(n_neurons=10000, dimensions=D, add_to_container=False)
    intercepts = ens.intercepts.sample(ens.n_neurons)
    plt.xlabel('Intercept')
    plot_intercept_distribution(ens)
    plt.title(f'{D} dimensions')
    plt.show()


# Activity analysis for redistributed neurons within an ensemble of 32 dimensions.def find_x_for_p(p, d):

def find_x_for_p(p, d):
    sign = 1
    if p > 0.5:
        p = 1.0 - p
        sign = -1
    return sign * np.sqrt(1 - scipy.special.betaincinv((d + 1) / 2.0, 0.5, 2 * p))

# Create a Nengo ensemble
ens = nengo.Ensemble(n_neurons=10000, dimensions=32, add_to_container=False)

# Sample intercepts
intercepts = ens.intercepts.sample(n=ens.n_neurons, d=1)[:, 0]

# Compute new intercepts
intercepts2 = [find_x_for_p(x_int / 2 + 0.5, ens.dimensions) for x_int in intercepts]
ens.intercepts = intercepts2

# Plot the new intercepts
plt.figure(figsize=(6, 4))
seaborn.distplot(intercepts2)
plt.xlabel('New intercepts')
plt.savefig('intercept_HD_mod.jpg', dpi=350)

# Plot the activity
plt.figure(figsize=(6, 4))
seaborn.distplot([find_x_for_p(x, ens.dimensions) for x in intercepts2])
plt.title('32 dimensions')
plt.xlabel('Activity')
plt.show()
