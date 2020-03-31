
pub use rand::random;

pub fn randf(n: f64) -> f64 {
    n*(2.*random::<f64>() - 1.)
}

pub fn randint(n: usize) -> usize {
    ((n as f64)*random::<f64>()) as usize
}

pub fn random_weight() -> f64 {
    randf(2.)
}

pub fn random_bias() -> f64 {
    randf(30.)
}

pub fn tanh(x: f64) -> f64 {
    x.tanh()
}