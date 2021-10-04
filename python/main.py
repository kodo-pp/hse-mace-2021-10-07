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

min_a = 2
max_a = 6
for a in np.linspace(2, 6, 11):
    params = sim.GameParameters(
        a=a,
        b=1,
        c=4,
        d=6,
        delta=3,
        alpha1=0,
        alpha2=0,
        beta1=0,
        beta2=0,
    )

    eq = params.compute_dynamics_params(epsilon=0.01).make_equation()
    #plot_cubic_parabola(eq)
    #print(eq)
    proc = sim.DynamicsProcess(eq, y0=1.0)
    conv = proc.test_convergence(40000, dt=0.001)
    print(f'{conv.alliance_failed} for a = {a}')
    history = conv.history()
    ys = [p.y for p in history[::]]
    ts = [p.t for p in history[::]]
    plt.plot(ts, ys, color=((a - min_a) / (max_a - min_a), 0.5, 0), linewidth=0.5)

plt.show()
