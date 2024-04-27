import numpy as np
import matplotlib.pyplot as plt

from neuron import h, gui


# Model definition by a ball and a stick

## Define the "Ball"
soma = h.Section(name='soma') 
soma.L = soma.diam = 12.6157    # [um]
soma.Ra = 100                   # Axial resistance [Ohm * cm]
soma.cm = 1                     # Membrane capacitance [uF / cm^2]
soma.insert('hh')               # Insert active Hodgkin-Huxley current in the soma
soma.gnabar_hh = 0.12           # Sodium conductance [S/cm2]
soma.gkbar_hh = 0.036           # Potassium conductance [S/cm2]
soma.gl_hh = 0.0003             # Leak conductance [S/cm2]
soma.el_hh = -54.3              # Reversal potential [mV]

## Define the "Stick"
dend = h.Section(name='dend')
dend.L = 200                    # [um]
dend.nseg = 101
dend.Ra = 100                   # Axial resistance [Ohm * cm]
dend.cm = 1                     # Membrane capacitance [uF / cm^2]
dend.diam = 1                   # [um]
dend.insert('pas')              # Insert passive current in the dendrite
dend.g_pas = 0.001              # Passive conductance [S/cm2] 
dend.e_pas = -65                # Leak reversal potential [mV]
dend.connect(soma(1)) 


# Define stimulation
stim = h.IClamp(dend(1))
stim.delay = 5
stim.dur = 1
stim_amp_array = [0.1, 0.3]


# Recording vectors
t_vec = h.Vector() 
t_vec.record(h._ref_t)
soma_v_vec = h.Vector()  
soma_v_vec.record(soma(0.5)._ref_v)
dend_v_vec_array = []
string_array = []
for i in np.flip(np.linspace(0, dend.L, 6)):
    dend_v_vec = h.Vector()
    string_array.append(f'{100 * (i / dend.L)} %')
    dend_v_vec.record(dend(i / dend.L)._ref_v)
    dend_v_vec_array.append(dend_v_vec)


# Simulation parameters
simdur = 25.0
h.tstop = simdur


# Simulation and plotting
for amp in stim_amp_array:
    stim.amp = amp
    h.run()
    
    cmap = plt.get_cmap('Blues')   
    colors = cmap(np.linspace(0,1,len(dend_v_vec_array) * 2))
    plt.figure(figsize=(8,4)) 

    plt.plot(t_vec, dend_v_vec_array[0], label=f'dendrite @ {string_array[0]}', linewidth=3, color=colors[-1]) 
    for i, v in enumerate(dend_v_vec_array[1:]):
        plt.plot(t_vec, v, label=f'dendrite @ {string_array[i + 1]}', color=colors[len(colors) - 2 - i]) 
    plt.plot(t_vec, soma_v_vec, label='soma', color='red', linewidth=3)
    plt.title('Cable Equation', fontsize=15)
    plt.xlim([5, 11])
    plt.xlabel('Time (ms)', fontsize=15) 
    plt.ylabel('Membrane Potential (mV)', fontsize=15) 
    plt.legend()
    plt.show()
    
h('forall {delete_section()}')
