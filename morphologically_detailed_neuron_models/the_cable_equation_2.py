import numpy as np
import matplotlib.pyplot as plt

from neuron import h, gui


soma = h.Section(name='soma') 
soma.L = soma.diam = 12.6157    # [um]
soma.Ra = 100                   # Axial resistance [Ohm * cm]
soma.cm = 1                     # Membrane capacitance [uF / cm^2]
soma.insert('hh')               # Insert active Hodgkin-Huxley current in the soma
soma.gnabar_hh = 0.12           # Sodium conductance [S/cm2]
soma.gkbar_hh = 0.036           # Potassium conductance [S/cm2]
soma.gl_hh = 0.0003             # Leak conductance [S/cm2]
soma.el_hh = -54.3              # Reversal potential [mV]

dend = h.Section(name='dend')
dend.L = 200                    # [um]
dend.Ra = 100                   # Axial resistance [Ohm * cm]
dend.cm = 1                     # Membrane capacitance [uF / cm^2]
dend.diam = 1                   # [um]
dend.insert('pas')              # Insert passive current in the dendrite
dend.g_pas = 0.001              # Passive conductance [S/cm2] 
dend.e_pas = -65                # Leak reversal potential [mV]
dend.connect(soma(1)) 

stim = h.IClamp(dend(1))
stim.delay = 5
stim.dur = 1
stim.amp = 0.3

resolution_array = [2, 4, 10]

t_vec = h.Vector() 
t_vec.record(h._ref_t)

soma_v_vec = h.Vector()  
soma_v_vec.record(soma(0.5)._ref_v)

dend_v_vec = h.Vector()
dend_v_vec.record(dend(1)._ref_v)
    
simdur = 25.0
h.tstop = simdur 

plt.figure(figsize=(10, 5))
line_types = [':', '--', '-']
for i, res in enumerate(resolution_array):
    dend.nseg = res
    h.finitialize(-65)
    h.run()
    plt.plot(t_vec, dend_v_vec, label=f'dendrite with {res} partitions', color='black', linewidth=3, linestyle=line_types[i]) 
    plt.plot(t_vec, soma_v_vec, label='soma', color='red', linewidth=3, linestyle=line_types[i])
    
plt.title('Cable Equation', fontsize=15)
plt.xlim([5, 11])
plt.xlabel('Time (ms)', fontsize=15) 
plt.ylabel('Membrane Potential (mV)', fontsize=15) 
plt.legend()
plt.show()
    
h('forall {delete_section()}')
