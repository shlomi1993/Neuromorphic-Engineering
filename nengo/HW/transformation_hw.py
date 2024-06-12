import nengo
import numpy as np
import matplotlib.pyplot as plt


T = 1.0
tau_synapse = 0.01

# Parameterization
neuron_count = [10, 100, 1000]
functions = [
    ('x^3', lambda x: x ** 3),
    ('sigmoid(x)', lambda x: 1 / (1 + np.exp(-x))),
    ('neg_relu(x)', lambda x: np.maximum(0, -x))
]

for func_name, func in functions:
    ensembles = dict()
    output_nodes = dict()
    probes = dict()

    # Model definition
    model = nengo.Network(label='Function Transformation')
    with model:
        input_node = nengo.Node(output=lambda t: 0.5 * np.sin(10 * t))
        for n_neurons in neuron_count:
            ensembles[n_neurons] = nengo.Ensemble(n_neurons=n_neurons, dimensions=1)
            output_nodes[n_neurons] = nengo.Node(size_in=1)
            nengo.Connection(input_node, ensembles[n_neurons])
            nengo.Connection(ensembles[n_neurons], output_nodes[n_neurons], function=func)
            probes[n_neurons] = nengo.Probe(output_nodes[n_neurons], synapse=tau_synapse)
        input_probe = nengo.Probe(input_node, synapse=tau_synapse)

    # Model simulation
    with nengo.Simulator(model) as sim:
        sim.run(T)

    # Plot results
    X = sim.trange()
    plt.figure(figsize=(18, 6))
    plt.suptitle('Function Transformation', fontsize=16, fontweight='bold')
    for i, n_neurons in enumerate(neuron_count, start=1):
        plt.subplot(1, 3, i)
        plt.title(f'Transformation from x to {func_name} using {n_neurons} Neurons', fontdict={'weight': 'bold'})
        plt.xlabel('x', fontdict={'weight': 'bold'}, labelpad=-2)
        plt.ylabel(func_name, fontdict={'weight': 'bold'}, labelpad=-5)
        plt.plot(X, sim.data[input_probe], label='Input')
        plt.plot(X, sim.data[probes[n_neurons]], label='Output')
        plt.plot(X, [func(x) for x in sim.data[input_probe]], label='Validation')
        plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'transform_{func_name}.png')
