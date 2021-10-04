use pyo3::class::PyObjectProtocol;
use pyo3::prelude::*;
use rand::{Rng, RngCore};
use rand_distr::StandardNormal;

#[pyclass]
#[derive(Debug, Copy, Clone)]
pub struct GameParameters {
    #[pyo3(get, set)]
    pub a: f64,
    #[pyo3(get, set)]
    pub b: f64,
    #[pyo3(get, set)]
    pub c: f64,
    #[pyo3(get, set)]
    pub d: f64,
    #[pyo3(get, set)]
    pub delta: f64,
    #[pyo3(get, set)]
    pub alpha1: f64,
    #[pyo3(get, set)]
    pub alpha2: f64,
    #[pyo3(get, set)]
    pub beta1: f64,
    #[pyo3(get, set)]
    pub beta2: f64,
}

#[pymethods]
impl GameParameters {
    #[new]
    pub fn new(
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        delta: f64,
        alpha1: f64,
        alpha2: f64,
        beta1: f64,
        beta2: f64,
    ) -> Self {
        Self {
            a,
            b,
            c,
            d,
            delta,
            alpha1,
            alpha2,
            beta1,
            beta2,
        }
    }

    pub fn compute_dynamics_params(&self, epsilon: f64) -> DynamicsParameters {
        let cap_a = self.b - self.a + self.f() + self.g();
        let cap_b = 2.0 * self.a - 2.0 * self.b - 2.0 * self.g() - self.f();
        let cap_c = self.g() + self.b - self.a;
        DynamicsParameters {
            cap_a,
            cap_b,
            cap_c,
            epsilon,
        }
    }

    pub fn inequity(&self) -> f64 {
        self.d - self.delta - self.a + self.c
    }

    pub fn f(&self) -> f64 {
        let inequity = self.inequity();
        let inequity_loss = if inequity > 0.0 {
            self.alpha1
        } else {
            -self.beta1
        } * inequity;
        self.a - self.c - inequity_loss
    }

    pub fn g(&self) -> f64 {
        let inequity = -self.inequity();
        let inequity_loss = if inequity > 0.0 {
            self.alpha2
        } else {
            -self.beta2
        } * inequity;
        self.d - self.delta - inequity_loss
    }
}

#[pyclass]
#[derive(Debug, Copy, Clone)]
pub struct DynamicsParameters {
    #[pyo3(get, set)]
    pub cap_a: f64,
    #[pyo3(get, set)]
    pub cap_b: f64,
    #[pyo3(get, set)]
    pub cap_c: f64,
    #[pyo3(get, set)]
    pub epsilon: f64,
}

#[pymethods]
impl DynamicsParameters {
    #[new]
    pub fn new(cap_a: f64, cap_b: f64, cap_c: f64, epsilon: f64) -> Self {
        Self {
            cap_a,
            cap_b,
            cap_c,
            epsilon,
        }
    }

    pub fn make_equation(&self) -> DynamicsEquation {
        let cubic_coef = 0.25 * self.cap_a;
        let quadratic_coef = 0.75 * self.cap_a + 0.5 * self.cap_b;
        let linear_coef = 0.75 * self.cap_a + self.cap_b + self.cap_c;
        let free_coef = 0.25 * self.cap_a + 0.5 * self.cap_b + self.cap_c;
        DynamicsEquation {
            cubic_coef,
            quadratic_coef,
            linear_coef,
            free_coef,
            epsilon: self.epsilon,
        }
    }

    pub fn cap_m(&self) -> f64 {
        -1.0 - 1.5 * self.cap_a / self.cap_b
    }

    pub fn alpha(&self) -> f64 {
        let cap_m = self.cap_m();
        let cap_a = self.cap_a;
        let cap_b = self.cap_b;
        let cap_c = self.cap_c;
        let minus_4_over_cap_a = -4.0 / cap_a;

        let horner = cap_a / 4.0;
        let horner = horner * cap_m + 0.75 * cap_a + 0.5 * cap_b;
        let horner = horner * cap_m * minus_4_over_cap_a + 0.75 * cap_a + cap_b + cap_c;
        let horner = horner * cap_m + 0.25 * cap_a + 0.5 * cap_b + cap_c;
        horner
    }

    pub fn beta(&self) -> f64 {
        let cap_m = self.cap_m();
        let cap_a = self.cap_a;
        let cap_b = self.cap_b;
        let cap_c = self.cap_c;
        let minus_4_over_cap_a = -4.0 / cap_a;

        let horner = 0.75;
        let horner = horner * cap_m + 1.5 * cap_a + cap_b;
        let horner = horner * cap_m + 0.75 * cap_a + cap_b + cap_c;
        horner * minus_4_over_cap_a
    }
}

#[pyclass]
#[derive(Debug, Copy, Clone)]
pub struct DynamicsEquation {
    #[pyo3(get, set)]
    pub cubic_coef: f64,
    #[pyo3(get, set)]
    pub quadratic_coef: f64,
    #[pyo3(get, set)]
    pub linear_coef: f64,
    #[pyo3(get, set)]
    pub free_coef: f64,
    #[pyo3(get, set)]
    pub epsilon: f64,
}

#[pymethods]
impl DynamicsEquation {
    #[new]
    pub fn new(
        cubic_coef: f64,
        quadratic_coef: f64,
        linear_coef: f64,
        free_coef: f64,
        epsilon: f64,
    ) -> Self {
        Self {
            cubic_coef,
            quadratic_coef,
            linear_coef,
            free_coef,
            epsilon,
        }
    }

    pub fn compute_dy(&self, y: f64, dt: f64, white_noise_value: f64) -> f64 {
        let horner = self.cubic_coef;
        let horner = horner * y + self.quadratic_coef;
        let horner = horner * y + self.linear_coef;
        let horner = horner * y + self.free_coef;
        horner * dt + self.epsilon * white_noise_value
    }
}

#[pyclass]
pub struct DynamicsProcess {
    equation: DynamicsEquation,
    t: f64,
    y: f64,
    rng: Box<dyn RngCore + Send>,
}

impl DynamicsProcess {
    pub fn new(equation: DynamicsEquation, y0: f64, rng: Box<dyn RngCore + Send>) -> Self {
        Self {
            equation,
            rng,
            t: 0.0,
            y: y0,
        }
    }
}

#[pymethods]
impl DynamicsProcess {
    #[new]
    pub fn with_thread_rng(equation: DynamicsEquation, y0: f64) -> Self {
        Self::new(equation, y0, Box::new(rand::rngs::OsRng))
    }

    pub fn step(&mut self, dt: f64) {
        let white_noise_value = self.rng.sample(StandardNormal);
        let dy = self.equation.compute_dy(self.y, dt, white_noise_value);
        self.y = (self.y + dy).clamp(-1.0, 1.0);
        self.t += dt;
        assert!(self.y.is_finite());
    }

    pub fn y(&self) -> f64 {
        self.y
    }

    pub fn t(&self) -> f64 {
        self.t
    }

    pub fn test_convergence(&mut self, num_steps: usize, dt: f64) -> ConvergenceResults {
        let mut history = Vec::with_capacity(num_steps + 1);
        for _ in 0..num_steps {
            history.push(HistoryPoint {
                t: self.t,
                y: self.y,
            });
            self.step(dt);
        }

        history.push(HistoryPoint {
            t: self.t,
            y: self.y,
        });
        let alliance_failed = self.y < 0.0;
        ConvergenceResults {
            history,
            alliance_failed,
        }
    }
}

impl std::fmt::Debug for DynamicsProcess {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            fmt,
            "DynamicsProcess {{ equation: {:?}, t: {}, y: {} }}",
            self.equation, self.t, self.y
        )
    }
}

#[pyclass]
#[derive(Debug, Copy, Clone)]
pub struct HistoryPoint {
    #[pyo3(get, set)]
    pub t: f64,
    #[pyo3(get, set)]
    pub y: f64,
}

#[pymethods]
impl HistoryPoint {
    #[new]
    pub fn new(t: f64, y: f64) -> Self {
        Self { t, y }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct ConvergenceResults {
    #[pyo3(get, set)]
    pub alliance_failed: bool,
    pub history: Vec<HistoryPoint>,
}

#[pymethods]
impl ConvergenceResults {
    #[new]
    pub fn new(alliance_failed: bool, history: Vec<HistoryPoint>) -> Self {
        Self { alliance_failed, history }
    }

    pub fn history(&self) -> Vec<HistoryPoint> {
        self.history.clone()
    }
}

macro_rules! impl_repr {
    ($class:ty) => {
        #[pyproto]
        impl<'p> PyObjectProtocol for $class {
            fn __repr__(&self) -> String {
                format!("{:?}", self)
            }
        }
    };
}

impl_repr!(GameParameters);
impl_repr!(DynamicsParameters);
impl_repr!(DynamicsEquation);
impl_repr!(DynamicsProcess);
impl_repr!(ConvergenceResults);
impl_repr!(HistoryPoint);

#[pymodule]
fn alliance_evolution_simulation(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<GameParameters>()?;
    module.add_class::<DynamicsParameters>()?;
    module.add_class::<DynamicsEquation>()?;
    module.add_class::<DynamicsProcess>()?;
    module.add_class::<ConvergenceResults>()?;
    module.add_class::<HistoryPoint>()?;
    Ok(())
}
