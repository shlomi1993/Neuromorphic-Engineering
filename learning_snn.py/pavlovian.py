import numpy as np
import matplotlib.pyplot as plt
import nengo


#  Pavlovian conditioning

D = 3
N = D * 100

def us_stim(t):
    """
    Cycle through the three US
    """
    t = t % 3
    if 0.9 < t < 1: return [1, 0, 0]
    if 1.9 < t < 2: return [0, 1, 0]
    if 2.9 < t < 3: return [0, 0, 1]
    return [0, 0, 0]

def cs_stim(t):
    """
    Cycle through the three CS
    """
    t = t % 3
    if 0.7 < t < 1: return [0.7, 0,   0.5]
    if 1.7 < t < 2: return [0.6, 0.7, 0.8]
    if 2.7 < t < 3: return [0, 1, 0]
    return [0, 0, 0]

def stop_learning(t):
    return 0 if 8 > t > 2 else 1

model = nengo.Network(label='Classical Conditioning')
with model:
    
    us_stim = nengo.Node(us_stim)
    us_stim_p = nengo.Probe(us_stim)
    
    us = nengo.Ensemble(N, D)
    ur = nengo.Ensemble(N, D)
    us_p = nengo.Probe(us, synapse=0.1)
    ur_p = nengo.Probe(ur, synapse=0.1)
    nengo.Connection(us, ur)
    nengo.Connection(us_stim, us[:D])
    
    cs_stim = nengo.Node(cs_stim)
    cs_stim_p = nengo.Probe(cs_stim)

    cs = nengo.Ensemble(N * 2, D * 2)
    cr = nengo.Ensemble(N, D)
    cs_p = nengo.Probe(cs, synapse=0.1)
    cr_p = nengo.Probe(cr, synapse=0.1)
    nengo.Connection(cs_stim, cs[:D])
    nengo.Connection(cs[:D], cs[D:], synapse=0.2)

    learn_conn = nengo.Connection(cs, cr, function=lambda x: [0]*D)
    learn_conn.learning_rule_type = nengo.PES(learning_rate=3e-4)

    error   = nengo.Ensemble(N, D)
    error_p = nengo.Probe(error, synapse=0.01)
    nengo.Connection(error, learn_conn.learning_rule)
    nengo.Connection(ur, error, transform=-1)
    nengo.Connection(cr, error, transform=1, synapse=0.1)

    stop_learn = nengo.Node(stop_learning)
    stop_learn_p = nengo.Probe(stop_learn)
    nengo.Connection(stop_learn, error.neurons, transform=-10 * np.ones((N, 1)))
    
with nengo.Simulator(model) as sim:
    sim.run(15)

t = sim.trange()
plt.figure(figsize=(12, 4))
plt.plot(t, sim.data[us_stim_p].T[0], c='blue', label='US #1')
plt.plot(t, sim.data[us_stim_p].T[1], c='red', label='US #2')
plt.plot(t, sim.data[us_stim_p].T[2], c='black', label='US #3')
plt.plot(t, sim.data[ur_p].T[0], c='blue', label='UR #1', linestyle=':', linewidth=3)
plt.plot(t, sim.data[ur_p].T[1], c='red', label='UR #2', linestyle=':', linewidth=3)
plt.plot(t, sim.data[ur_p].T[2], c='black', label='UR #3', linestyle=':', linewidth=3)
plt.ylabel('Value')
plt.xlabel('Time (sec)')
plt.legend()
plt.show()
plt.figure(figsize=(12, 4))
plt.plot(t, sim.data[cs_stim_p].T[0], c='blue', label='CS #1')
plt.plot(t, sim.data[cs_stim_p].T[1], c='red', label='CS #2')
plt.plot(t, sim.data[cs_stim_p].T[2], c='black', label='CS #3')
plt.plot(t, sim.data[ur_p].T[0], c='blue', label='UR #1', linestyle=':', linewidth=3)
plt.plot(t, sim.data[ur_p].T[1], c='red', label='UR #2', linestyle=':', linewidth=3)
plt.plot(t, sim.data[ur_p].T[2], c='black', label='UR #3', linestyle=':', linewidth=3)
plt.ylabel('Value')
plt.xlabel('Time (sec)')
plt.legend()
plt.show()
plt.figure(figsize=(12, 4))
plt.plot(t, sim.data[cs_stim_p].T[0], c='blue', label='CS #1')
plt.plot(t, sim.data[cs_stim_p].T[1], c='red', label='CS #2')
plt.plot(t, sim.data[cs_stim_p].T[2], c='black', label='CS #3')
plt.plot(t, sim.data[cr_p].T[0], c='blue', label='CR #1', linestyle=':', linewidth=3)
plt.plot(t, sim.data[cr_p].T[1], c='red', label='CR #2', linestyle=':', linewidth=3)
plt.plot(t, sim.data[cr_p].T[2], c='black', label='CR #3', linestyle=':', linewidth=3)
plt.ylabel('Value')
plt.xlabel('Time (sec)')
plt.legend()
plt.show()
plt.figure(figsize=(12, 4))
plt.plot(t, sim.data[error_p].T[0], c='black', label='error')
plt.ylabel('Value')
plt.xlabel('Time (sec)')
plt.legend()
plt.show()
plt.figure(figsize=(12, 2))
plt.plot(t, sim.data[stop_learn_p].T[0], c='black',  label='stop')
plt.ylabel('Value')
plt.xlabel('Time (sec)')
plt.legend()
plt.show()
