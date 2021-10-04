#!/usr/bin/env python
import alliance_evolution_simulation as sim
import matplotlib.pyplot as plt
import numpy as np

def plot_cubic_parabola(eq: sim.DynamicsEquation):
    xs = np.linspace(-1.5, 1.5, 1000)
    dys = [eq.compute_dy(x, 1, 0) for x in xs]
    plt.plot(xs, dys)
    plt.plot([-1.5, 1.5], [0, 0], 'red')
    plt.plot([-1.0, -1.0], [min(dys) - 1, max(dys) + 1], 'red')
    plt.plot([1.0, 1.0], [min(dys) - 1, max(dys) + 1], 'red')
    plt.show()


for a in np.linspace(6.5, 8.5, 10):
    params = sim.GameParameters(
        a=a,
        b=2,
        c=7,
        d=3.5,
        delta=1,
        alpha1=0.15,
        alpha2=0.15,
        beta1=0.2,
        beta2=0.2,
    )

    eq = params.compute_dynamics_params(epsilon=3).make_equation()
    plot_cubic_parabola(eq)
    #print(eq)
    proc = sim.DynamicsProcess(eq, y0=1.0)
    conv = proc.test_convergence(10000, dt=0.001)
    print(f'{conv.alliance_failed} for a = {a}')
    history = conv.history()
    ts = [p.t for p in history]
    ys = [p.y for p in history]
    #plt.plot(ts, ys)

plt.show()
