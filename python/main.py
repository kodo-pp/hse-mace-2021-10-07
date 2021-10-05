#!/usr/bin/env python
import alliance_evolution_simulation as sim
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from tqdm import trange
from typing import Optional, Callable, Tuple, Dict, Any, NoReturn

def plot_cubic_parabola(eq: sim.DynamicsEquation) -> None:
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

@dataclass
class ParamValueTuple:
    a: float
    b: float
    c: float
    d: float
    delta: float
    alpha1: float
    alpha2: float
    beta1: float
    beta2: float

    def as_array(self) -> np.array:
        return np.array([
            self.a,
            self.b,
            self.c,
            self.d,
            self.delta,
            self.alpha1,
            self.alpha2,
            self.beta1,
            self.beta2,
        ])

    def __len__(self) -> int:
        return 9

def catastrophe_scatterplot(
    num_points: int,
    reference_params: ParamValueTuple,
    noise_scale: ParamValueTuple,
    epsilon: float,
    num_steps: int,
    dt: float,
    y0: float = -1.0,
    alpha_outlier_threshold: float = 50,
    beta_outlier_threshold: float = 50,
    random_distribution: Callable[[Tuple[int, int]], None] = np.random.standard_normal,
    plot_alpha: float = 0.25,
    plot_marker: Any = '.',
) -> None:
    outcomes = []
    params_noise = random_distribution((num_points, len(reference_params)))
    scaled_params_noise = params_noise @ np.diagflat(noise_scale.as_array())
    params_array = reference_params.as_array() + scaled_params_noise
    num_outliers = 0

    for i in trange(num_points):
        a, b, c, d, delta, alpha1, alpha2, beta1, beta2 = params_array[i, :]
        params = sim.GameParameters(
            a=a,
            b=b,
            c=c,
            d=d,
            delta=delta,
            alpha1=alpha1,
            alpha2=alpha2,
            beta1=beta1,
            beta2=beta2,
        )

        dynamics_params = params.compute_dynamics_params(epsilon)
        alpha = dynamics_params.alpha()
        beta = dynamics_params.beta()

        if abs(alpha) > alpha_outlier_threshold or abs(beta) > beta_outlier_threshold:
            # Outliers spoil the plot, remove them.
            num_outliers += 1
            continue

        proc = sim.DynamicsProcess(dynamics_params.make_equation(), y0=y0)
        alliance_failed = proc.test_convergence(num_steps=num_steps, dt=dt).alliance_failed
        outcomes.append((alpha, beta, alliance_failed))

    print(
        f'Outliers filtered out: {num_outliers}',
        f'({round(num_outliers/num_points*100, 3)}%)',
    )
    outcomes = np.array(outcomes)
    plt.scatter(
        outcomes[:, 0],
        outcomes[:, 1],
        color=['red' if failed else 'green' for failed in outcomes[:, 2]],
        marker=plot_marker,
        alpha=plot_alpha,
    )

def do_scatter(num_points) -> None:
    catastrophe_scatterplot(
        num_points=num_points,
        reference_params=ParamValueTuple(
            a=4.8,
            b=1.5,
            c=4.0,
            d=8.0,
            delta=3.0,
            alpha1=0.0,
            alpha2=0.0,
            beta1=0.0,
            beta2=0.0,
        ),
        noise_scale=ParamValueTuple(
            a=0.8,
            b=0.3,
            c=0.1,
            d=0.3,
            delta=0.1,
            alpha1=0.0,
            alpha2=0.0,
            beta1=0.0,
            beta2=0.0,
        ),
        epsilon=0.01,
        num_steps=4000,
        dt=0.003,
        y0=-1.0,
        beta_outlier_threshold=3,
    )
    plt.xlabel(r'$\alpha$')
    # Thanks to https://stackoverflow.com/a/27671918.
    plt.ylabel(r'$\beta$', rotation=0)
    # … as well as to
    # https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html.
    plt.legend(
        [mpatches.Patch(color='green'), mpatches.Patch(color='red')],
        ['Alliance didn\'t fail', 'Alliance failed'],
    )
    plt.show()

def parse_args() -> Namespace:
    ap = ArgumentParser()
    sub = ap.add_subparsers(required=True, dest='command')

    sub_scatter = sub.add_parser('scatter', help='Draw a catastrophe scatter plot.')
    sub_scatter.add_argument('num_points', type=int, help='Number of points to consider.')

    return ap.parse_args()

def main() -> None:
    args = parse_args()
    if args.command == 'scatter':
        do_scatter(num_points=args.num_points)

if __name__ == '__main__':
    main()
