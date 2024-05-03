import numpy as np
import matplotlib.pyplot as plt

# Model Parameters
sim_time = 50 * 1e-3                    # Simulation time          [seconds]
dt = 0.1 * 1e-3                         # Simulation time interval [seconds]
V_rest = -70 * 1e-3                     # Resting potential        [V]
R_m = 1 * 1e3                           # Membrane Resistance      [Ohm]
C_m = 5 * 1e-6                          # Membrane Capacitance     [F]
tau_ref = 1 * 1e-3                      # Refractory Period        [seconds]
V_spike = 50 * 1e-3                     # Spike voltage            [V]
T = np.arange(0, sim_time + dt, dt)     # Time array               [list of seconds]

# Define the stimulus as a list of 'flat' current inputs
I = np.array([1e-04] * len(T))

# Plot current dynamic
plt.figure(figsize=(10, 5))
plt.title(f'Current Dynamic - Intensity over time', fontsize=15) 
plt.xlabel('Time (msec)', fontsize=15)
plt.ylabel('Current Intensity (uA)', fontsize=15)
plt.plot(T * 1e+3, I * 1e+6, color='blue', linewidth=2)
plt.show()

# Simulate for each threshold value
for V_th_mV in [-70, -30, 10]:

    # Simulation parameters
    t_init = 0                          # Stimulus init time       [seconds]
    V_m = np.ones(len(T)) * V_rest      # Membrane voltage array   [list of V]
    V_th = V_th_mV * 1e-3               # Spike threshold          [V]
    tau = R_m * C_m                     # Membrane time constant   [seconds]
    spikes = []                         # Spikes timings           [list of milliseconds]
    F = [0]                             # Frequencies              [list of Hz]

    # Simulation:
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
        freq = 1 / (spikes[-1] - spikes[-2]) if len(spikes) > 1 else 0
        F.append(freq) 

        msg = f"step={i}, t={t * 1e+3} ms, f={freq} Hz, I(t)={I[i] * 1e+6} uA, u_inf={(V_rest + R_m * I[i])} V, u_now={V_m[i]} V"
        if spiked:
            msg += ", SPIKED!"

        print(msg)

    # Plot simulation results
    plt.figure(figsize=(10, 5))
    plt.title(f'Leaky Integrate-and-Fire Model (V_th={V_th} V)', fontsize=15) 
    plt.ylabel('Membrane Potential (mV)', fontsize=15) 
    plt.xlabel('Time (msec)', fontsize=15)
    plt.plot(T * 1e3, V_m * 1e3, linewidth=5, label='V_m')
    plt.plot(T * 1e3, 100 / max(I) * I, label='Stimuli (Scaled)', color='sandybrown', linewidth=2)
    plt.ylim([-75, 120])
    plt.axvline(x=spikes[0], c='red', label='Spike')
    for s in spikes[1:]:
        plt.axvline(x=s, c='red')
    plt.axhline(y=V_th / 1e-3, c='black', label = 'Threshold', linestyle='--')
    plt.legend()
    plt.show()
