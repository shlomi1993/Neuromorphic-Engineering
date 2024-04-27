import numpy as np
import matplotlib.pyplot as plt


for tau_factor in [6]:

    # Model Parameters:
    T = 50 * 1e-3                   # Simulation time          [Sec]
    dt = 0.1 * 1e-3                 # Simulation time interval [Sec]
    t_init = 0                      # Stimulus init time       [V]
    V_rest = -70 * 1e-3             # Resting potential        [V]
    R_m = 1 * 1e3                   # Membrane Resistance      [Ohm]
    C_m = 5 * 1e-6                  # Capacitance              [F]
    tau_ref = 1 * 1e-3              # Refractory Period        [Sec]
    V_th = -40 * 1e-3               # Spike threshold          [V]
    dI = 0.5 * 1e-6                       # Current intensity step   [A]
    V_spike = 50 * 1e-3             # Spike voltage            [V]

    # Simulation parameters:
    T = np.arange(0, T + dt, dt)    # Time array
    V_m = np.ones(len(T)) * V_rest  # Membrane voltage array
    tau = R_m * C_m * tau_factor    # Time constant 
    spikes = []                     # Spikes timings
    F = [0]                         # Frequencies

    # Define the stimulus:
    I = np.array([dI * t for t in range(len(T))])

    # Simulation:
    for i, t in enumerate(T[:-1]):
        freq = 1 / (spikes[-1] - spikes[-2]) if len(spikes) > 1 else 0
        F.append(freq) 
        if t > t_init:
            V_m_inf_i = V_rest + R_m * I[i]
            V_m[i + 1] = V_m_inf_i + (V_m[i] - V_m_inf_i) * np.exp(-dt / tau)
            if V_m[i] >= V_th:
                spikes.append(t * 1e3)
                V_m[i] = V_spike
                t_init = t + tau_ref
        print(f"step={i}, t={t}, f={freq}, I(t)={I[i]}, u_inf={(V_rest + R_m * I[i])}, u_now={V_m[i]}")

    # Plot:
    plt.figure(figsize=(10, 5))
    plt.title(f'Leaky Integrate-and-Fire Model (tau={tau:.2f})', fontsize=15) 
    plt.xlabel('Stimuli Current Intensity (A)', fontsize=15)
    plt.ylabel('Spike Frequency (1 / ms)', fontsize=15)
    plt.plot(I, F, color='sandybrown', linewidth=2)
    plt.show()
