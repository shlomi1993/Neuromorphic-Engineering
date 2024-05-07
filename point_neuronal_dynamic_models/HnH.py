import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from uuid import uuid4


@dataclass
class Gate:
    '''
    This class represents a gate in an Hodgkin-Huxley circuit
    '''
    alpha: float = 0
    beta: float = 0
    state: float = 0

    def update(self, delta_tms: float) -> None:
        alpha_state = self.alpha * (1 - self.state)
        beta_state = self.beta * self.state
        self.state += delta_tms * (alpha_state - beta_state)

    def set_infinite_state(self) -> None:
        self.state = self.alpha / (self.alpha + self.beta)


@dataclass
class HhModelResults:
    times: np.ndarray
    stimuli: np.ndarray
    V_m: np.ndarray
    n: np.ndarray
    m: np.ndarray
    h: np.ndarray
    I_Na: np.ndarray
    I_K: np.ndarray
    I_leak: np.ndarray
    I_sum: np.ndarray

    def plot(self, membrane_potential: bool = False, gate_states: bool = False, ion_currents: bool = False,
             to_file: bool = False) -> None:
        assert membrane_potential or gate_states or ion_currents, 'No plot options selected'

        # Plot membrane potential over time
        if membrane_potential:
            plt.figure(figsize=(10,5))
            plt.plot(self.times, self.V_m - 70, linewidth=2, label='Vm')
            plt.plot(self.times, self.stimuli - 70, label='Stimuli (Scaled)', linewidth=2, color='sandybrown')
            plt.ylabel('Membrane Potential (mV)', fontsize=15)
            plt.xlabel('Time (msec)', fontsize=15)
            plt.xlim([90, 160])
            plt.title('Hodgkin-Huxley Neuron Model', fontsize=15)
            plt.legend(loc=1)
            plt.savefig(f'hh_membrane_potential_{uuid4()}.png') if to_file else plt.show()

        # Plot gate states over time
        if gate_states:
            plt.figure(figsize=(10,5))
            plt.plot(self.times, self.m, label='m (Na)', linewidth=2)
            plt.plot(self.times, self.h, label='h (Na)', linewidth=2)
            plt.plot(self.times, self.n, label='n (K)', linewidth=2)
            plt.ylabel('Gate state', fontsize=15)
            plt.xlabel('Time (msec)', fontsize=15)
            plt.xlim([90, 160])
            plt.title('Hodgkin-Huxley Spiking Neuron Model: Gatings', fontsize=15)
            plt.legend(loc=1)
            plt.savefig(f'hh_gate_states_{uuid4()}.png') if to_file else plt.show()


        # Plot ion currents over time
        if ion_currents:
            plt.figure(figsize=(10,5))
            plt.plot(self.times, self.I_Na, label='I_Na', linewidth=2)
            plt.plot(self.times, self.I_K, label='I_K', linewidth=2)
            plt.plot(self.times, self.I_leak, label='I_leak', linewidth=2)
            plt.plot(self.times, self.I_sum, label='I_sum', linewidth=2)
            plt.ylabel('Current (uA)', fontsize=15)
            plt.xlabel('Time (msec)', fontsize=15)
            plt.title('Hodgkin-Huxley Spiking Neuron Model: Ion Currents', fontsize=15)
            plt.xlim([90, 160])
            plt.legend(loc=1)
            plt.savefig(f'hh_ion_currents_{uuid4()}.png') if to_file else plt.show()


class HhModel:
    '''
    This class implements the Hodgkin-Huxley model
    '''
    def __init__(self, starting_voltage: float = 0.0, membrane_capacitance: float = 1.0, E_Na: float = 115.0,
                 E_K: float = -12.0, E_leak: float = 10.6, g_Na: float = 120, g_K: float = 36, g_leak: float = 0.3) -> None:
        # Initialize model parameters
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_leak = E_leak
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_leak = g_leak
        self.V_m = starting_voltage
        self.C_m = membrane_capacitance

        # Initialize gates
        self.m = Gate()
        self.n = Gate()
        self.h = Gate()
        self.update_gate_time_constants(starting_voltage)
        self.m.set_infinite_state()
        self.n.set_infinite_state()
        self.h.set_infinite_state()

        # Initialize currents
        self.I_Na = 0.0
        self.I_K  = 0.0
        self.I_leak = 0.0
        self.I_sum = 0.0

    def update_gate_time_constants(self, V_m: float) -> None:
        self.n.alpha = .01 * ((10 - V_m) / (np.exp((10 - V_m) / 10) - 1))
        self.n.beta = .125 * np.exp(-V_m / 80)
        self.m.alpha = .1 * ((25 - V_m) / (np.exp((25 - V_m) / 10) - 1))
        self.m.beta = 4 *np.exp(-V_m / 18)
        self.h.alpha = .07 * np.exp(-V_m / 20)
        self.h.beta = 1 / (np.exp((30 - V_m) / 10) + 1)

    def update_cell_voltage(self, stimulus_current: float, delta_tms: float) -> None:
        self.I_Na = np.power(self.m.state, 3) * self.g_Na * self.h.state*(self.V_m - self.E_Na)
        self.I_K = np.power(self.n.state, 4) * self.g_K * (self.V_m - self.E_K)
        self.I_leak = self.g_leak * (self.V_m - self.E_leak)
        self.I_sum = stimulus_current - self.I_Na - self.I_K - self.I_leak
        self.V_m += delta_tms * self.I_sum / self.C_m

    def update_gate_states(self, delta_tms: float) -> None:
        self.n.update(delta_tms)
        self.m.update(delta_tms)
        self.h.update(delta_tms)

    def _iterate(self, stimulus_current: float = 0.0, delta_tms: float = 0.05) -> float:
        self.update_gate_time_constants(self.V_m)
        self.update_cell_voltage(stimulus_current, delta_tms)
        self.update_gate_states(delta_tms)

    def simulate(self, point_count: int) -> HhModelResults:
        V_m = np.empty(point_count)
        n = np.empty(point_count)
        m = np.empty(point_count)
        h = np.empty(point_count)
        I_Na = np.empty(point_count)
        I_K = np.empty(point_count)
        I_leak = np.empty(point_count)
        I_sum = np.empty(point_count)
        times = np.arange(point_count) * 0.05
        stimuli = np.zeros(point_count)
        stimuli[2000:3000] = 10

        for i in range(len(times)):
            self._iterate(stimulus_current=stimuli[i], delta_tms=0.05)
            V_m[i] = self.V_m
            n[i]  = self.n.state
            m[i]  = self.m.state
            h[i]  = self.h.state
            I_Na[i] = self.I_Na
            I_K[i] = self.I_K
            I_leak[i] = self.I_leak
            I_sum[i] = self.I_sum
            
        return HhModelResults(times, stimuli, V_m, n, m, h, I_Na, I_K, I_leak, I_sum)
        

def main():
    E_assignments = [
        # {'E_Na': 115, 'E_K': -12, 'E_leak': 10.6},    # Standard values
        {'E_Na': 40,  'E_K': -12, 'E_leak': 10.6},      # Low E_Na
        # {'E_Na': 100, 'E_K': -12, 'E_leak': 10.6},
        # {'E_Na': 190, 'E_K': -12, 'E_leak': 10.6},
        # {'E_Na': 100, 'E_K': -45, 'E_leak': 10.6},
        # {'E_Na': 100, 'E_K': -30, 'E_leak': 10.6},
        # {'E_Na': 100, 'E_K': 5,   'E_leak': 10.6},
        {'E_Na': 100, 'E_K': 10,  'E_leak': 10.6},      # High E_K
        {'E_Na': 115, 'E_K': -12, 'E_leak': -20},       # Low E_leak
        # {'E_Na': 115, 'E_K': -12, 'E_leak': 350},
    ]

    for E_values in E_assignments:
        print(E_values)
        hh = HhModel(**E_values)
        results = hh.simulate(point_count=5000)
        results.plot(membrane_potential=True, to_file=True)


if __name__ == '__main__':
    main()
