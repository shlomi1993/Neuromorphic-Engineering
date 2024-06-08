import numpy as np
import matplotlib.pyplot as plt


#  Differentiable LIF tuning curve

Rm = 1000  # Membrane Resistance  [kOhm]
Cm = 5e-6  # Capacitance          [uF]

t_ref = 10e-3
tau = Rm * Cm
v_th = 1
I0 = 1
R = 1000
la = 0.05

rho  = lambda x: np.max(x,0)
rho2 = lambda x: la * np.log(1 + np.exp(x / la))

a3  = lambda i: 1 / (t_ref + tau * np.log(1 + (v_th / rho(i - v_th))))
a4  = lambda i: 1 / (t_ref + tau * np.log(1 + (v_th / rho2(i - v_th))))

I = np.linspace(0, 3, 10000)
A3 = [a3(i) for i in I]
A4 = [a4(i) for i in I]

plt.plot(I,A3, linewidth=3, color='black', label=r'$\rho=max(x,0)$')
plt.plot(I,A4, linewidth=3, color='red', label=r'$\rho=\lambda log(1+e^{x/\lambda})$')
plt.ylim(0,100)
plt.xlim(0,3)
plt.legend()
plt.ylabel("Firing rate (Hz)")
plt.xlabel("Input current")
plt.show()
