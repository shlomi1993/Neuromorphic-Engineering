import numpy as np
import matplotlib.pyplot as plt

from scipy import signal


# Model Parameters:
T = 50 * 1e-3                   # Simulation time          [Sec]
dt = 0.1 * 1e-3                 # Simulation time interval [Sec]
t_init = 0                      # Stimulus init time       [V]
V_rest = -70 * 1e-3             # Resting potential        [V]
R_m = 1 * 1e3                   # Membrane Resistance      [Ohm]
C_m = 5 * 1e-6                  # Capacitance              [F]
tau_ref = 1 * 1e-3              # Refractory Period        [Sec]
v_th = -40 * 1e-3               # Spike threshold          [V]
I = 0.2 * 1e-3                  # Current stimulus         [A]
V_spike = 50 * 1e-3             # Spike voltage            [V]

# Simulation parameters:
T = np.arange(0, T + dt, dt)    # Time array
V_m = np.ones(len(T)) * V_rest  # Membrane voltage array
tau = R_m * C_m                 # Time constant 
spikes = []                     # Spikes timings

# Define the stimulus:
I = I * signal.windows.triang(len(T))  # Triangular stimulation pattern

# Simulation:
# import ipdb; ipdb.set_trace(context=11)
for i, t in enumerate(T[:-1]):
    if t > t_init:
        V_m_inf_i = V_rest + R_m * I[i]
        V_m[i + 1] = V_m_inf_i + (V_m[i] - V_m_inf_i) * np.exp(-dt / tau)
        if V_m[i] >= v_th:
            spikes.append(t * 1e3) 
            V_m[i] = V_spike
            t_init = t + tau_ref

# Plot:
plt.figure(figsize=(10, 5))
plt.title('Leaky Integrate-and-Fire Model', fontsize=15) 
plt.ylabel('Membrane Potential (mV)', fontsize=15) 
plt.xlabel('Time (msec)', fontsize=15)
plt.plot(T * 1e3, V_m * 1e3, linewidth=5, label='V_m')
plt.plot(T * 1e3, 100 / max(I) * I, label='Stimuli (Scaled)', color='sandybrown', linewidth=2)
plt.ylim([-75, 100])
plt.axvline(x=spikes[0], c='red', label='Spike')
for s in spikes[1:]:
    plt.axvline(x=s, c='red')
plt.axhline(y=v_th / 1e-3, c='black', label = 'Threshold', linestyle='--')
plt.legend()
plt.show()
