import numpy as np
import matplotlib.pyplot as plt

from neuron import h, gui


# Model creation

cells = {}

for i in range(3):

    soma = h.Section(name='soma') 
    soma.L = soma.diam = 12.6157    # [um]
    soma.Ra = 100                   # Axial resistance [Ohm * cm]
    soma.cm = 1                     # Membrane capacitance [uF / cm^2]
    soma.insert('hh')               # Insert active Hodgkin-Huxley current in the soma
    soma.gnabar_hh = 0.12           # Sodium conductance [S / cm2]
    soma.gkbar_hh = 0.036           # Potassium conductance [S / cm2]
    soma.gl_hh = 0.0003             # Leak conductance [S / cm2]
    soma.el_hh = -54.3              # Reversal potential [mV]

    dend = h.Section(name='dend')
    dend.L = 200                    # [um]
    dend.nseg = 101
    dend.Ra = 100                   # Axial resistance [Ohm * cm]
    dend.cm = 1                     # Membrane capacitance [uF / cm^2]
    dend.diam = 1                   # [um]
    dend.insert('pas')              # Insert passive current in the dendrite
    dend.g_pas = 0.001              # Passive conductance [S / cm2] 
    dend.e_pas = -65                # Leak reversal potential [mV]
    dend.connect(soma(1)) 
    
    cells[i] = {'soma': soma, 'dend': dend}

syns    = [h.ExpSyn(cells[1]['dend'](0.5)), h.ExpSyn(cells[2]['dend'](0.5))]
netcons = [h.NetCon(cells[0]['soma'](0.5)._ref_v, syns[0], sec=cells[0]['soma']), 
           h.NetCon(cells[1]['soma'](0.5)._ref_v, syns[1], sec=cells[1]['soma'])]

for netcon in netcons:
    netcon.weight[0] = 0.04
    netcon.delay = 5

syn_ = h.ExpSyn(cells[0]['dend'](0.5))


# Stimulation

stim = h.NetStim()
stim.number = 1
stim.start = 5
ncstim = h.NetCon(stim, syn_)
ncstim.delay = 1
ncstim.weight[0] = 0.04
stim_amp_array = [0.08]


# Recording vectors

t_vec = h.Vector() 
t_vec.record(h._ref_t)

for cell in cells:
    cells[cell]['soma_Vm'] = h.Vector()
    cells[cell]['soma_Vm'].record(cells[cell]['soma'](0.5)._ref_v)

dend_v_vec_array = []
string_array     = []

for i in np.flip(np.linspace(0, dend.L, 6)):
    dend_v_vec = h.Vector()
    percent = 100 * (i / cells[0]['dend'].L)
    string_array.append(f'{percent} %')
    dend_v_vec.record(cells[0]['dend'](i/cells[0]['dend'].L)._ref_v)
    dend_v_vec_array.append(dend_v_vec)


# Simulation parameters

simdur = 40
h.tstop = simdur 


# run simulation

h.run()


# Plot

cmap = plt.get_cmap('Blues')   
colors = cmap(np.linspace(0, 1, len(dend_v_vec_array) * 2))
plt.figure(figsize=(8, 4)) 

plt.plot(t_vec, dend_v_vec_array[0], linewidth=3, color=colors[-1]) 
for i, vec in enumerate(dend_v_vec_array[1:]):
    plt.plot(t_vec, vec, color = colors[len(colors)-2-i]) 

plt.plot(t_vec, cells[0]['soma_Vm'], label='soma @ cell 1', color='red', linewidth=3)
plt.plot(t_vec, cells[1]['soma_Vm'], label='soma @ cell 2', color='green', linewidth=3)
plt.plot(t_vec, cells[2]['soma_Vm'], label='soma @ cell 3', color='orange', linewidth=3)
plt.title("Compartmental Model", fontsize=15)
plt.xlabel('Time (ms)', fontsize=15) 
plt.ylabel("Membrane Potential (mV)", fontsize=15) 
plt.legend()
plt.show()
    
h("forall {delete_section()}")
