import numpy as np
import matplotlib.pyplot as plt


# Model and simulation parameters for three response dynamics:

x = 5
y = 140

titles = ['Regular Spiking', 'Chattering', 'Fast spiking']
A = [0.02, 0.02, 0.1]
B = [0.2, 0.2 , 0.2]
C = [-65, -50, -65]
D = [8, 2, 2]

v0 = -70    # Resting potential        [mV]
T = 200     # Simulation time          [mSec]
dt = 0.25   # Simulation time interval [mSec]

time  = np.arange(0, T + dt, dt)  # Time array


# Define the stimulus a step function:

stimulus = np.zeros(len(time))
stimulus[20:] = 15


# Simulation:

trace = np.zeros((2, len(time)))  # Tracing du and dv
for a, b, c, d in zip(A, B, C, D):
    v  = v0
    u  = b * v
    spikes = []
    for i, stimulus_value in enumerate(stimulus):
        v += dt * (0.04 * v ** 2 + x * v + y - u + stimulus_value) 
        u += dt * a * (b * v - u)
        if v > 30:
            trace[0, i] = 30
            v = c 
            u += d
        else:
            trace[0, i] = v 
            trace[1, i] = u


    # Plot:
    plt.figure(figsize=(10, 5))
    plt.title(f'Izhikevich Model: {titles}', fontsize=15) 
    plt.ylabel('Membrane Potential (mV)', fontsize=15) 
    plt.xlabel('Time (msec)', fontsize=15)
    plt.plot(time, trace[0], linewidth=2, label='Vm')
    plt.plot(time, trace[1], linewidth=2, label='Recovery', color='green')
    plt.plot(time, stimulus + v0, label = 'Stimuli (Scaled)', color='sandybrown', linewidth=2)
    plt.legend(loc=1)
    plt.show()
