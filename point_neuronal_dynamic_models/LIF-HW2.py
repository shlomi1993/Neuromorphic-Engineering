import numpy as np
import matplotlib.pyplot as plt


for V_th_mV in [-70, -30, 10]:

    # Model Parameters:
    T = 50 * 1e-3                   # Simulation time          [Sec]
    dt = 0.1 * 1e-3                 # Simulation time interval [Sec]
    t_init = 0                      # Stimulus init time       [V]
    V_rest = -70 * 1e-3             # Resting potential        [V]
    R_m = 1 * 1e3                   # Membrane Resistance      [Ohm]
    C_m = 5 * 1e-6                  # Capacitance              [F]
    tau_ref = 1 * 1e-3              # Refractory Period        [Sec]
    V_th = V_th_mV * 1e-3           # Spike threshold          [V]
    dI = 0.5 * 1e-6                 # Current intensity step   [A]
    V_spike = 50 * 1e-3             # Spike voltage            [V]

    # Simulation parameters:
    T = np.arange(0, T + dt, dt)    # Time array
    V_m = np.ones(len(T)) * V_rest  # Membrane voltage array
    tau = R_m * C_m                 # Time constant 
    spikes = []                     # Spikes timings
    F = [0]                         # Frequencies

    # Define the stimulus:
    I = np.array([1e-04] * len(T))  # 'Flat' input current stimuli

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
        freq = 1 / (spikes[-1] - spikes[-2]) if len(spikes) > 1 else 0
        F.append(freq) 
        msg = f"step={i}, t={t}, f={freq}, I(t)={I[i]}, u_inf={(V_rest + R_m * I[i])}, u_now={V_m[i]}"
        if spiked:
            msg += ", SPIKED!"
        print(msg)

    # Plot:
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

