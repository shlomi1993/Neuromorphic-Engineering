import numpy as np
import matplotlib.pyplot as plt

# Model Parameters
sim_time = 50 * 1e-3                    # Simulation time          [seconds]
dt = 0.1 * 1e-3                         # Simulation time interval [seconds]
V_rest = -70 * 1e-3                     # Resting potential        [V]
R_m = 1 * 1e3                           # Membrane Resistance      [Ohm]
C_m = 5 * 1e-6                          # Membrane Capacitance     [F]
tau_ref = 1 * 1e-3                      # Refractory Period        [seconds]
V_th = -40 * 1e-3                       # Spike threshold          [V]
V_spike = 50 * 1e-3                     # Spike voltage            [V]
dI = 0.5 * 1e-6                         # Current intensity step   [A]
T = np.arange(0, sim_time + dt, dt)     # Time array               [list of seconds]

# Define the stimulus as a list of current inputs
I = np.array([dI * t for t in range(len(T))])

# Plot current dynamic
plt.figure(figsize=(10, 5))
plt.title(f'Current Dynamic - Intensity over time', fontsize=15) 
plt.xlabel('Time (msec)', fontsize=15)
plt.ylabel('Current Intensity (uA)', fontsize=15)
plt.plot(T * 1e+3, I * 1e+6, color='blue', linewidth=2)
plt.show()

# Simulate for each tau
for tau_factor in [6]:

    # Simulation parameters
    t_init = 0                          # Stimulus init time       [seconds]
    V_m = np.ones(len(T)) * V_rest      # Membrane voltage array   [list of V]
    tau = R_m * C_m * tau_factor        # Membrane time constant   [seconds]
    spikes = []                         # Spikes timings           [list of milliseconds]
    F = [0]                             # Frequencies              [list of Hz]

    # Simulation
    for i, t in enumerate(T[:-1]):
        spiked = False
        if t > t_init:
            V_m_inf_i = V_rest + R_m * I[i]
            V_m[i + 1] = V_m_inf_i + (V_m[i] - V_m_inf_i) * np.exp(-dt / tau)
            if V_m[i] >= V_th:
                spiked = True
                spikes.append(t * 1e3)
                V_m[i] = V_spike
                t_init = t + tau_ref

        # Calculate frequency as 1 divided by time-cycle, where the time-cycle is defined as the time between spikes]
        freq = 1 / (spikes[-1] - spikes[-2]) if len(spikes) > 1 else 0  # Hz
        F.append(freq) 

        msg = f"step={i}, t={t * 1e+3} ms, f={freq} Hz, I(t)={I[i] * 1e+6} uA, u_inf={(V_rest + R_m * I[i])} V, u_now={V_m[i]} V"
        if spiked:
            msg += ", SPIKED!"

        print(msg)

    # Plot simulation results
    plt.figure(figsize=(10, 5))
    plt.title(f'Leaky Integrate-and-Fire Model (tau={tau:.2f})', fontsize=15) 
    plt.xlabel('Stimuli Current Intensity (uA)', fontsize=15)
    plt.ylabel('Spike Frequency (Hz)', fontsize=15)
    plt.plot(I * 1e+6, F, color='sandybrown', linewidth=2)
    plt.show()
