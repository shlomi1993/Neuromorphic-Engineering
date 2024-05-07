import numpy as np
import matplotlib.pyplot as plt


a = 0.1
b = 0.26
c = -65
d = 2

v0 = -70
T = 200
dt = 0.1
time = np.arange(0, T + dt, dt)  # Time array

stimuli = np.zeros(len(time)) - 4
stimuli[301:] += 5
stimuli[1350:1371] += 5

trace = np.zeros((2, len(time))) # Tracing du and dv

v  = v0
u  = b * v
for i, I in enumerate(stimuli):
    v += dt * (0.04 * v ** 2 + 5 * v + 140 - u + I)
    u += dt * a * (b * v - u)
    if v > 30:
        trace[0, i] = 30
        v = c
        u += d
    else:
        trace[0, i] = v 
        trace[1, i] = u

plt.figure(figsize=(10, 5))
plt.title('Izhikevich Model: RZ', fontsize=15) 
plt.ylabel('Membrane Potential (mV)', fontsize=15) 
plt.xlabel('Time (msec)', fontsize=15)
plt.plot(time, trace[0], linewidth=2, label='Vm')
plt.plot(time, trace[1], linewidth=2, label='Recovery', color='green')
plt.plot(time, stimuli + v0, label='Stimuli (Scaled)', color='sandybrown', linewidth=2)
plt.legend(loc=1)
plt.show()
