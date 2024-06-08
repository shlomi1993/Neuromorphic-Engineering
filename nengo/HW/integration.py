model = nengo.Network()
with model:
    input_node = nengo.Node(lambda t: np.sin(t))
    
    # Single neuron ensemble
    ensemble_1_neuron = nengo.Ensemble(1, dimensions=1)
    
    # Large ensemble for integrator
    ensemble_integrator = nengo.Ensemble(100, dimensions=1)
    
    # Connecting input to single neuron
    nengo.Connection(input_node, ensemble_1_neuron)
    
    # Connecting input to integrator with feedback for integration
    nengo.Connection(input_node, ensemble_integrator)
    nengo.Connection(ensemble_integrator, ensemble_integrator, synapse=0.1)
    
    # Probes to record data
    probe_1_neuron = nengo.Probe(ensemble_1_neuron, synapse=0.01)
    probe_integrator = nengo.Probe(ensemble_integrator, synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(5.0)

# Plotting the results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.title("Single Neuron Output")
plt.plot(sim.trange(), sim.data[probe_1_neuron])
plt.xlabel("Time (s)")
plt.ylabel("Output")

plt.subplot(2, 1, 2)
plt.title("Integrator Output")
plt.plot(sim.trange(), sim.data[probe_integrator])
plt.xlabel("Time (s)")
plt.ylabel("Output")

plt.show()
