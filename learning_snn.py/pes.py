import numpy as np
import matplotlib.pyplot as plt
import nengo


# PES learning: communication channel

model = nengo.Network('Learn a Communication Channel')
with model:
    stim = nengo.Node(output=nengo.processes.WhiteSignal(10, high=5, rms=0.5))
    
    pre = nengo.Ensemble(n_neurons=60, dimensions=1)
    post = nengo.Ensemble(n_neurons=60, dimensions=1)
    
    nengo.Connection(stim, pre)
    conn = nengo.Connection(pre, post, function=lambda x: np.random.random())
    
    inp_p = nengo.Probe(stim)
    pre_p = nengo.Probe(pre, synapse=0.01)
    post_p = nengo.Probe(post, synapse=0.01)
    
    error = nengo.Ensemble(n_neurons=60, dimensions=1)
    error_p = nengo.Probe(error, synapse=0.03)
    
    nengo.Connection(post, error)
    nengo.Connection(pre, error, transform=-1)  # Learn simple communication line
    conn.learning_rule_type = nengo.PES()
    learn_conn = nengo.Connection(error, conn.learning_rule) 

with nengo.Simulator(model) as sim:
    sim.run(10.0)

t = sim.trange()
plt.figure(figsize=(12, 4))
plt.plot(t, sim.data[inp_p].T[0], c='k', label='Input')
plt.plot(t, sim.data[pre_p].T[0], c='b', label='Pre')
plt.plot(t, sim.data[post_p].T[0], c='r', label='Post')
plt.ylabel('Value')
plt.legend(loc=1)
plt.figure(figsize=(12, 4))
plt.plot(t, sim.data[error_p].T[0], c='k', label='Error')
plt.ylabel('Value')
plt.xlabel('Time (sec)')
plt.legend(loc='best')
plt.show()
