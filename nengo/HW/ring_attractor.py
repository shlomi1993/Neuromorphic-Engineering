import nengo
import numpy as np
import matplotlib.pyplot as plt

# Define tau synapse constant
tau_synapse = 0.01

# Define input point
input_point = [1.0, 0.0, -1.0]

# Define the parameters for the ring attractor model
omega_x = 1.0
omega_y = 1.2
omega_z = 0.8

# Define the recurrent connection function
def ring_attractor(state):
    x, y, z = state
    new_x = x + (omega_z * z - omega_y * y)
    new_y = y + (omega_x * x - omega_z * z)
    new_z = z + (omega_y * y - omega_x * x)
    return new_x, new_y, new_z

# Model definition
model = nengo.Network()
with model:

    # Set the original ring attractor initial conditions as initial stimulus that "turned off" after tau_synapse seconds
    input_node = nengo.Node(lambda t: input_point if t <= tau_synapse else [0.0, 0.0, 0.0])  

    # Represents a 3D state using 3000 neurons to achieve accuracy at a tolerable computational cost
    ensemble = nengo.Ensemble(n_neurons=3000, dimensions=3, radius=2.0)  

    # Connect the input node to the ensemble, and the ensemble to itself using the ring attractor function
    nengo.Connection(input_node, ensemble, synapse=tau_synapse)
    nengo.Connection(ensemble, ensemble, function=ring_attractor, synapse=tau_synapse)

    # Attach a probe to the ensemble to measure state values
    probe_ensemble = nengo.Probe(ensemble, synapse=tau_synapse)

# Model simulation
with nengo.Simulator(model) as sim:
    sim.run(1.1)

# Extract the solution trajectories
x_values = sim.data[probe_ensemble][:, 0]
y_values = sim.data[probe_ensemble][:, 1]
z_values = sim.data[probe_ensemble][:, 2]

# Find the required starting point index
start_index = 0
target = np.array(input_point)
closest_distance = float('inf')
for i, (x, y, z) in enumerate(zip(x_values, y_values, z_values), 1):
    current_point = np.array([x, y, z])
    distance = np.linalg.norm(current_point - target)
    if distance < closest_distance:
        closest_distance = distance
        start_index = i

# Ignore the data collected during the model stabilization time
x_solution = x_values[start_index:]
y_solution = y_values[start_index:]
z_solution = z_values[start_index:]

# Plotting the trajectories in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_solution, y_solution, z_solution, label='Ring Attractor Trajectory')
ax.scatter(x_solution[0], y_solution[0], z_solution[0], color='red', label='Start')
ax.scatter(x_solution[-1], y_solution[-1], z_solution[-1], color='green', label='End')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Ring Attractor Simulation')
ax.legend()
plt.show()
