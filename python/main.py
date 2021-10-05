#!/usr/bin/env python
import alliance_evolution_simulation as sim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

def plot_cubic_parabola(eq: sim.DynamicsEquation):
    xs = np.linspace(-1.5, 1.5, 1000)
    dys = [eq.compute_dy(x, 1, 0) for x in xs]
    plt.plot(xs, dys)
    plt.plot([-1.5, 1.5], [0, 0], 'red')
    plt.plot([-1.0, -1.0], [min(dys) - 1, max(dys) + 1], 'red')
    plt.plot([1.0, 1.0], [min(dys) - 1, max(dys) + 1], 'red')
    plt.show()

#min_a = 2
#max_a = 6
#for a in np.linspace(2, 6, 11):
#    params = sim.GameParameters(
#        a=a,
#        b=1,
#        c=4,
#        d=6,
#        delta=3,
#        alpha1=0,
#        alpha2=0,
#        beta1=0,
#        beta2=0,
#    )
#
#    eq = params.compute_dynamics_params(epsilon=0.01).make_equation()
#    #plot_cubic_parabola(eq)
#    #print(eq)
#    proc = sim.DynamicsProcess(eq, y0=1.0)
#    conv = proc.test_convergence(40000, dt=0.001)
#    print(f'{conv.alliance_failed} for a = {a}')
#    history = conv.history()
#    ys = [p.y for p in history[::]]
#    ts = [p.t for p in history[::]]
#    plt.plot(ts, ys, color=((a - min_a) / (max_a - min_a), 0.5, 0), linewidth=0.5)
#
#plt.show()

outcomes = []
num_points = 10000
ref_params = np.array([4.8, 1.5, 4, 8, 3])
params_array = ref_params + np.random.normal(
    size=(num_points, len(ref_params)),
    loc=0,
    scale=1,
) @ np.diagflat([0.8, 0.3, 0.1, 0.3, 0.1])

for i in trange(num_points):
    a, b, c, d, delta = params_array[i, :]
    params = sim.GameParameters(
        a=a,
        b=b,
        c=c,
        d=d,
        delta=delta,
        alpha1=0.0,
        alpha2=0.0,
        beta1=0.0,
        beta2=0.0,
    )

    dyn_params = params.compute_dynamics_params(epsilon=0.01)
    alpha = dyn_params.alpha()
    beta = dyn_params.beta()

    if abs(alpha) > 50 or abs(beta) > 50:
        # Outliers spoil the plot, remove them.
        continue
    proc = sim.DynamicsProcess(dyn_params.make_equation(), y0=-1.0)
    alliance_failed = proc.test_convergence(4000, dt=0.003).alliance_failed
    outcomes.append((alpha, beta, alliance_failed))

outcomes = np.array(outcomes)
plt.scatter(
    outcomes[:, 0],
    outcomes[:, 1],
    color=['red' if failed else 'green' for failed in outcomes[:, 2]],
    marker='.',
    alpha=0.3,
)
plt.show()

#arr = np.array(outcomes[False])
#arr = arr[:, arr[1, :] < 0.426]
#print(arr)
#print(arr.shape)
