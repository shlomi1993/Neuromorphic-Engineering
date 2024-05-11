import sys
import re
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

    def __init__(self, T: float, dt: float, v_0: float = -70, v_apex: float = 30, stabilization_time: float = 100) -> None:
        """
        Initializes the Izhikevich model with the given parameters.

        Args:
            T (float): Simulation total time                                [milliseconds]
            dt (float): Simulation time interval                            [milliseconds]
            v_0 (float): Membrane resting potential                         [milliseconds]
            v_spike_apex (float): Spike voltage apex                        [mV]
            stimulus (np.ndarray): Stimulus current intensities             [Array of Amperes]
            stabilization_time (float): model voltage stabilization time    [milliseconds]
        """
        assert T >= 0 and stabilization_time >= 0, 'times cannot be negative'
        self.set_time(T, dt, stabilization_time)
        self.v_0 = v_0
        self.v_spike_apex = v_apex

    @property
    def times(self):
        return self._times
    
    @property
    def dt(self):
        return self._dt

    def set_time(self, T: float, dt: float, stabilization_time: float = 100) -> None:
        self._times = np.arange(-stabilization_time, T + dt, dt)   # Time array
        self._dt = dt
        self.start_idx = int(stabilization_time / self._dt)  # Experiment start time by index, after model stabilization

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

        trace = np.zeros((2, len(self.times)))  # For tracing du and dv
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

    def plot(self, title: str, stimulus: np.ndarray, trace: np.ndarray, savefig: bool = False) -> None:
        """
        Plots the membrane potential over time as simulated by the model, using the given simulation trace and stimulus.

        Args:
            title (str): The title of the plot.
            stimulus (np.ndarray): The simulation input stimulus current array.
            trace (np.ndarray): The simulation result's trace array, which contains the v and u values over time.
            savefig (bool): Whether to save the plot to a file.

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
        plt.plot(self.times[self.start_idx:], trace[0][self.start_idx:], linewidth=2, label='Vm')
        plt.plot(self.times[self.start_idx:], trace[1][self.start_idx:], linewidth=2, label='Recovery', color='green')
        plt.plot(self.times[self.start_idx:], (stimulus + self.v_0)[self.start_idx:], label='Stimuli (Scaled)', color='sandybrown', linewidth=2)
        plt.legend(loc=1)
        if savefig:
            clean_title = re.sub(r'\s*\([^()]*\)\s*', '', title).replace(' ', '_').replace('-', '_')
            plt.savefig(fname=clean_title + '.png')
        else:
            plt.show()


def main():
    savefig = len(sys.argv) > 1 and sys.argv[1] == '--savefig'

    # Instantiate an Izhikevich model
    izhikevich = IzhikevichModel(T=200, dt=0.1)

    # Define stimulus as a step function
    step_stimulus = np.zeros(len(izhikevich.times))
    step_stimulus[izhikevich.start_idx + 100:] = 10

    # Define a stimulus as a step function with an additional pulse after a while for the Resonator (RZ) dynamic
    step_pulse_stimulus = np.zeros(len(izhikevich.times))
    step_pulse_stimulus[izhikevich.start_idx + 100:] = 0.2
    step_pulse_stimulus[izhikevich.start_idx + 1000:izhikevich.start_idx + 1020] = 5

    # Define a stimulus as a negative step function for the second Thalamo-Cortical (TC) dynamic
    neg_step_stimulus = np.zeros(len(izhikevich.times)) - 10
    neg_step_stimulus[izhikevich.start_idx + 100:] = 0

    # Simulation parameters for six response dynamics, and title for each experiment
    experiments = [
        ('Regular Spiking (RS)',         -70,  IzhikevichParams(a=0.02, b=0.2,  c=-65, d=8),     step_stimulus),
        ('Intrinsically Bursting (IB)',  -70,  IzhikevichParams(a=0.02, b=0.2,  c=-55, d=4),     step_stimulus),
        ('Chattering (CH)',              -70,  IzhikevichParams(a=0.02, b=0.2,  c=-50, d=2),     step_stimulus),
        ('Fast Spiking (FS)',            -70,  IzhikevichParams(a=0.1,  b=0.2,  c=-65, d=2),     step_stimulus),
        ('Low Threshold Spiking (LTS)',  -70,  IzhikevichParams(a=0.02, b=0.25, c=-65, d=2),     step_stimulus),
        ('Resonator (RZ)',               -70,  IzhikevichParams(a=0.1,  b=0.26, c=-65, d=2),     step_pulse_stimulus),
        ('Thalamo-Cortical (TC)',        -63,  IzhikevichParams(a=0.02, b=0.25, c=-65, d=0.05),  step_stimulus),
        ('Thalamo-Cortical (TC) - Neg',  -87,  IzhikevichParams(a=0.02, b=0.25, c=-65, d=0.05),  neg_step_stimulus)
    ]

    # Simulate and plot each experiment
    for title, v_0, params, stimulus in experiments:
        izhikevich.v_0 = v_0
        trace = izhikevich.simulate(params, stimulus)
        izhikevich.plot(title, stimulus, trace, savefig)


if __name__ == '__main__':
    main()
