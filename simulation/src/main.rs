use rand::Rng;
use rand_distr::StandardNormal;

#[derive(Debug, Copy, Clone)]
pub struct GameParameters {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub delta: f64,
    pub alpha1: f64,
    pub alpha2: f64,
    pub beta1: f64,
    pub beta2: f64,
}

impl GameParameters {
    pub fn compute_rde_params(&self, epsilon: f64) -> DynamicsParameters {
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

#[derive(Debug, Copy, Clone)]
pub struct DynamicsParameters {
    pub cap_a: f64,
    pub cap_b: f64,
    pub cap_c: f64,
    pub epsilon: f64,
}

impl DynamicsParameters {
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
}

#[derive(Debug, Copy, Clone)]
pub struct DynamicsEquation {
    pub cubic_coef: f64,
    pub quadratic_coef: f64,
    pub linear_coef: f64,
    pub free_coef: f64,
    pub epsilon: f64,
}

impl DynamicsEquation {
    pub fn compute_dy(&self, y: f64, dt: f64, white_noise_value: f64) -> f64 {
        let horner = self.cubic_coef;
        let horner = horner * y + self.quadratic_coef;
        let horner = horner * y + self.linear_coef;
        let horner = horner * y + self.free_coef;
        (horner + self.epsilon * white_noise_value) * dt
    }
}

#[derive(Debug, Copy, Clone)]
pub struct DynamicsProcess<R: Rng> {
    equation: DynamicsEquation,
    t: f64,
    y: f64,
    rng: R,
}

impl<R: Rng> DynamicsProcess<R> {
    pub fn new(equation: DynamicsEquation, y0: f64, rng: R) -> Self {
        Self {
            equation,
            rng,
            t: 0.0,
            y: y0,
        }
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

    pub fn test_convergence(mut self, num_steps: usize, dt: f64) -> ConvergenceResults {
        let mut history = Vec::with_capacity(num_steps + 1);
        for _ in 0..num_steps {
            history.push(HistoryPoint { t: self.t, y: self.y });
            self.step(dt);
        }

        history.push(HistoryPoint { t: self.t, y: self.y });
        let alliance_failed = self.y < 0.0;
        ConvergenceResults { history, alliance_failed }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct HistoryPoint {
    pub t: f64,
    pub y: f64,
}

#[derive(Debug, Clone)]
pub struct ConvergenceResults {
    pub alliance_failed: bool,
    pub history: Vec<HistoryPoint>,
}

fn main() {
    for a in (600..800).map(|x| x as f64 / 100.0) {
        let initial_params = GameParameters {
            a,
            b: 2.0,
            c: 7.0,
            d: 3.5,
            delta: 1.0,
            alpha1: 0.15,
            alpha2: 0.15,
            beta1: 0.2,
            beta2: 0.2,
        };

        let eq = initial_params.compute_rde_params(0.1).make_equation();
        let proc = DynamicsProcess::new(eq, 1.0, rand::thread_rng());
        let results = proc.test_convergence(10000, 0.01);
        print!("a = {:5.2} :: ", a);
        if results.alliance_failed {
            println!("\x1b[31mAlliance failed\x1b[0m");
        } else {
            println!("\x1b[32mAlliance stays strong\x1b[0m");
        }
    }
}
