import nengo
import numpy as np
import matplotlib.pyplot as plt


T = 1.0
tau_synapse = 0.01

# Parameterization
neuron_count = [10, 100, 1000]
functions = [
    ('$x^3$', lambda x: x ** 3),
    ('$\sigma(x)$', lambda x: 1 / (1 + np.exp(-x))),
    ('$max(0, -x)$', lambda x: np.maximum(0, -x))
]

for func_name, func in functions:
    for n_neurons in neuron_count:

        # Model definition
        model = nengo.Network(label='Function Transformation')
        with model:
            input_node = nengo.Node(output=lambda t: 0.5 * np.sin(10 * t))
            ensemble = nengo.Ensemble(n_neurons=n_neurons, dimensions=1)
            output_node = nengo.Node(size_in=1)
            nengo.Connection(input_node, ensemble)
            nengo.Connection(ensemble, output_node, function=func)
            input_probe = nengo.Probe(input_node, synapse=tau_synapse)
            output_probe = nengo.Probe(output_node, synapse=tau_synapse)
            

        # Model simulation
        with nengo.Simulator(model) as sim:
            sim.run(T)

        # Plot results
        t = sim.trange()
        plt.figure(figsize=(8, 5))
        plt.title(f'Transformation from $x$ to {func_name} using {n_neurons} Neurons', fontdict={'size': 14})
        plt.xlabel('Time (s)', fontdict={'size': 12}, labelpad=-1)
        plt.ylabel('Value', fontdict={'size': 12}, labelpad=-4)
        plt.plot(t, sim.data[input_probe], label='$x$')
        plt.plot(t, sim.data[output_probe], label=f'{func_name} using {n_neurons} neurons')
        plt.plot(t, [func(x) for x in sim.data[input_probe]], label=f'Validation for {func_name}')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        clean_func_name = func_name.replace('$', '').replace('\\', '')
        plt.savefig(f'transform_{clean_func_name}_{n_neurons}.png')
