import numpy as np
import matplotlib.pyplot as plt
import nengo

from nengo.utils.ensemble import tuning_curves


subplot_title_fontdict = {'size': 16, 'weight': 'bold'}
axis_title_fontdict = {'size': 14, 'weight': 'bold'}
axis_values_fontdict = {'size': 12, 'weight': 'bold'}

target = lambda x: x + np.sin(x)
tau_synapse = 0.01


# Simulate for each requested number of neurons
for n_neurons in [1, 10, 50, 1000]:

    # Model definition
    model = nengo.Network()
    with model:
        input_node = nengo.Node(target)
        ensemble = nengo.Ensemble(n_neurons=n_neurons, dimensions=1, radius=30, seed=1)
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
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(f'Representation of f(x)=x+sin(x) using {n_neurons} neuron{"s" if n_neurons > 1 else ""}',
                 fontsize=22, fontweight='bold')

    ## Plot tuning curves
    plt.subplot(1, 2, 1)
    plt.title('Tuning Curves', fontdict=subplot_title_fontdict)
    plt.xlabel('I (mA)', fontdict=axis_title_fontdict)
    plt.ylabel('a (Hz)', fontdict=axis_title_fontdict)
    plt.plot(*tuning_curves(ensemble, sim))

    ## Plot function representation
    plt.subplot(1, 2, 2)
    plt.title('Function Representation', fontdict=subplot_title_fontdict)
    plt.xlabel('x', fontdict=axis_title_fontdict)
    plt.ylabel('f(x)', fontdict=axis_title_fontdict)
    plt.plot(t, sim.data[input_probe],'r', linewidth=6)
    plt.plot(t, sim.data[output_probe])

    ## Display plot
    # plt.show()
    plt.savefig(f'tuning_curves_{n_neurons}.png')
