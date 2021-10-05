from typing import List

class GameParameters:
    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        d: float,
        delta: float,
        alpha1: float,
        alpha2: float,
        beta1: float,
        beta2: float,
    ):
        ...

    def compute_dynamics_params(self, epsilon: float) -> 'DynamicsParameters':
        ...

    def inequity(self) -> float:
        ...

    def f(self) -> float:
        ...

    def g(self) -> float:
        ...

    a: float
    b: float
    c: float
    d: float
    delta: float
    alpha1: float
    alpha2: float
    beta1: float
    beta2: float

class DynamicsParameters:
    def __init__(
        self,
        cap_a: float,
        cap_b: float,
        cap_c: float,
        epsilon: float,
    ) -> None:
        ...

    def make_equation(self) -> 'DynamicsEquation':
        ...

    def alpha(self) -> float:
        ...

    def beta(self) -> float:
        ...

    def cap_m(self) -> float:
        ...

    cap_a: float
    cap_b: float
    cap_c: float
    epsilon: float


class DynamicsEquation:
    def __init__(
        self,
        cubic_coef: float,
        quadratic_coef: float,
        linear_coef: float,
        free_coef: float,
        epsilon: float,
    ) -> None:
        ...

    def compute_dy(self, y: float, dt: float, white_noise_value: float) -> float:
        ...

    cubic_coef: float
    quadratic_coef: float
    linear_coef: float
    free_coef: float
    epsilon: float

class DynamicsProcess:
    def __init__(self, equation: 'DynamicsEquation', y0: float, clamp: bool) -> None:
        ...

    def step(self, dt: float) -> None:
        ...

    def y(self) -> float:
        ...

    def t(self) -> float:
        ...

    def test_convergence(self, num_steps: int, dt: float) -> 'ConvergenceResults':
        ...

class ConvergenceResults:
    def __init__(self, alliance_failed: bool, history: List['HistoryPoint']) -> None:
        ...

    def history(self) -> List['HistoryPoint']:
        ...

    alliance_failed: bool

class HistoryPoint:
    def __init__(self, t: float, y: float) -> None:
        ...

    t: float
    y: float

def run_convertence_test_for_all(
    eqs: List['DynamicsEquation'],
    num_steps: int,
    dt: float,
    y0: float,
    clamp: bool,
) -> List['ConvergenceResults']:
    ...
