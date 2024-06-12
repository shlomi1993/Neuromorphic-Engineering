import numpy as np
import matplotlib.pyplot as plt
import nengo

from nengo.utils.matplotlib import rasterplot
from nengo.utils.ensemble import tuning_curves


target = lambda x: x + np.sin(x)
tau_synapse = 0.01

# Simulate for each requested number of neurons
for n_neurons in [100]:#[1, 10, 50, 1000]:

    # Model definition
    model = nengo.Network()
    with model:
        input_node = nengo.Node(target)
        ensemble = nengo.Ensemble(n_neurons=n_neurons, dimensions=1, radius=30)
        output_node = nengo.Node(size_in=1)
        nengo.Connection(input_node, ensemble)
        nengo.Connection(ensemble, output_node)

        input_probe = nengo.Probe(input_node, synapse=tau_synapse)
        output_probe = nengo.Probe(output_node, 'output', synapse=tau_synapse)

    # Model simulation
    with nengo.Simulator(model) as sim:
        sim.run(20.0)

    # Plot results
    t = sim.trange()
    fig = plt.figure(figsize=(24, 6))

    ## Plot tuning curves
    plt.subplot(1, 3, 1)
    plt.title(f'{ensemble.n_neurons} Neuron Tuning Curves')
    plt.xlabel('I (mA)')
    plt.ylabel('a (Hz)')
    plt.plot(*tuning_curves(ensemble, sim))

    ## Plot encoding
    plt.subplot(1, 3, 2)
    plt.title(f'Encoding by {ensemble.n_neurons} Neurons')
    plt.xlabel('Time')
    plt.ylabel('Encoded value')
    plt.plot(t, sim.data[input_probe],'r', linewidth=4)
    plt.ax = plt.gca()
    rasterplot(t, sim.data[output_probe], ax=plt.ax.twinx(), use_eventplot=True)

    ## Plot decoding
    plt.subplot(1, 3, 3)
    plt.title(f'Decoding by {ensemble.n_neurons} Neurons')
    plt.xlabel('time (s)')
    plt.ylabel('Encoded value')
    plt.plot(t, sim.data[input_probe],'r', linewidth=4)
    plt.plot(t, sim.data[output_probe])

    ## Display plot
    fig.tight_layout(pad=5.0)
    plt.show()
