import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass


@dataclass
class IzhikevichParams:
    a: float    # Describes the time-scale of the recovery variable u - smaller values result in slower recovery
    b: float    # Describes the sensitivity of the recovery variable u to the sub-threshold fluctuations of the membrane potential v
    c: float    # Describes the after-spike reset value of the membrane potential v
    d: float    # Describes the after-spike reset value of the recovery variable u

    def as_tuple(self):
        return tuple(vars(self).values())


class IzhikevichModel:

    def __init__(self, T: float, dt: float, v_0: float, v_apex: float) -> None:
        """
        Initializes the Izhikevich model with the given parameters.

        Args:
            T (float): Simulation total time                    [milliseconds]
            dt (float): Simulation time interval                [milliseconds]
            v_0 (float): Membrane resting potential             [milliseconds]
            v_spike_apex (float): Spike voltage apex            [mV]
            stimulus (np.ndarray): Stimulus current intensities [Array of Amperes]
        """
        self.times = np.arange(0, T + dt, dt)   # Time array
        self.dt = dt
        self.v_0 = v_0
        self.v_spike_apex = v_apex

    def simulate(self, params: IzhikevichParams, stimulus: np.ndarray) -> np.ndarray:
        """
        Simulates the Izhikevich model with the given parameters and input stimulus currents.

        Args:
            params (IzhikevichParams): simulation a, b, c, and d parameters
            stimulus (np.ndarray): Stimulus current intensities [Array of Amperes]

        Returns:
            trace (np.ndarray): Tracing du and dv
        """
        a, b, c, d = params.as_tuple()

        trace = np.zeros((2, len(self.times)))   # For tracing du and dv
        v  = self.v_0                           # v represents the memory potential in mV
        u  = b * v                              # u represents the membrane recovery variable

        for i, I in enumerate(stimulus):
            v += self.dt * (0.04 * v ** 2 + 5 * v + 140 - u + I) 
            u += self.dt * a * (b * v - u)
            if v > self.v_spike_apex:
                trace[0, i] = self.v_spike_apex
                v = c
                u += d
            else:
                trace[0, i] = v 
                trace[1, i] = u

        return trace

    def plot(self, title: str, stimulus: np.ndarray, trace: np.ndarray) -> None:
        """
        Plots the membrane potential over time as simulated by the model, using the given simulation trace and stimulus.

        Args:
            title (str): The title of the plot.
            stimulus (np.ndarray): The simulation input stimulus current array.
            trace (np.ndarray): The simulation result's trace array, which contains the v and u values over time.

        Raises:
            ValueError: If the shapes of the stimulus and trace arrays are incompatible.

        Note:
            The trace array should have shape (2, N), where N is the number of time steps, with the first row
            representing the values of membrane potential and the second row representing the values of the recovery variable.
        """
        plt.figure(figsize=(10, 5))
        plt.title(f'Izhikevich Model: {title}', fontsize=15) 
        plt.ylabel('Membrane Potential (mV)', fontsize=15) 
        plt.xlabel('Time (msec)', fontsize=15)
        plt.plot(self.times, trace[0], linewidth=2, label='Vm')
        plt.plot(self.times, trace[1], linewidth=2, label='Recovery', color='green')
        plt.plot(self.times, stimulus + self.v_0, label='Stimuli (Scaled)', color='sandybrown', linewidth=2)
        plt.legend(loc=1)
        plt.show()


def main():

    # Instantiate an Izhikevich model
    izhikevich = IzhikevichModel(T=200, dt=0.1, v_0=-70, v_apex=30)

    # Define the stimulus a step function
    stimulus = np.zeros(len(izhikevich.times))
    stimulus[41:] = 10

    # Simulation parameters for six response dynamics, and title for each experiment
    params_and_titles = [
        (IzhikevichParams(a=0.02, b=0.2, c=-65, d=8), 'Regular Spiking (RS)'),
        (IzhikevichParams(a=0.02, b=0.2, c=-55, d=4), 'Intrinsically Bursting (IB)'),
        (IzhikevichParams(a=0.02, b=0.2, c=-50, d=2), 'Chattering (CH)'),
        (IzhikevichParams(a=0.1,  b=0.2, c=-65, d=2), 'Fast Spiking (FS)'),
        (IzhikevichParams(a=0.1, b=0.26, c=-65, d=5), 'Resonator (RZ)'),  # TODO
        (IzhikevichParams(a=0.02, b=0.25, c=-65, d=2), 'Low Threshold Spiking (LTS)'),
    ]

    # Simulate and plot
    for params, title in params_and_titles:
        trace = izhikevich.simulate(params, stimulus)
        izhikevich.plot(title=title, stimulus=stimulus, trace=trace)

    # TODO
    # Handle the Thalamo-Cortical cases by modifying the membrane resting potential value
    # params = IzhikevichParams(a=0.02, b=0.25, c=-65, d=0.05)
    # for v_0 in [-63, -87]:
    #     izhikevich.v_0 = v_0
    #     trace = izhikevich.simulate(params, stimulus)
    #     izhikevich.plot(title=f'Thalamo-Cortical (TC) when v_0={v_0}', stimulus=stimulus, trace=trace)


if __name__ == '__main__':
    main()
