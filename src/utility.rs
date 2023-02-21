#![allow(non_snake_case)]
#![allow(dead_code)]

use rand::distributions::Distribution;
use rand::Rng;

use crate::linalg;

// TBD; use "traits"?
//
// i.e.,
//
//      U = - a * p + W @ V, a ~ LogNormal(ma, s), W ~ Normal(mw, S)
//
// for some mean vectors ma, mw (Jx1), variances s (Jx1) and covariance S (IxK).
// Or,
//
//      U = a * log(b - p) + W @ V
//      U = - a / (b-p) + W @ V
//
// etc.
//
// The main things we need here are:
//
//  1. sampling a, b, W
//  2. supplying bimax, bmax (for corrected methods); probably return from sampling
//  3. a function to compute U, DpU, and maybe DppU given p
//
// Note that 3 should probably populate passed mutable references.
//
pub trait Utility {
    /// Sample whatever utility needs
    fn sample(&mut self, I: usize) -> Option<(usize, f64)>; // return bimax, bmax || inf, inf

    /// Compute the pre-computable, non-price portion of utility
    fn values(&self, J: usize, X: &[f64], V: &mut [f64]);

    /// Evaluate price component of utility with no derivatives
    fn eval_UpD0(&self, p: &[f64], U: &mut [f64]);

    /// Evaluate price component of utility with first derivatives
    fn eval_UpD1(&self, p: &[f64], U: &mut [f64], DpU: &mut [f64]);

    /// Evaluate price component of utility with second derivatives
    fn eval_UpD2(&self, p: &[f64], U: &mut [f64], DpU: &mut [f64], DppU: &mut [f64]);
}

/// A linear-in-price utility
pub struct LinUtility<'u, AD: Distribution<f64>, WD: Distribution<f64>> {
    I: usize,
    K: usize,
    a: Vec<f64>, // I vector
    // no b (budgets) required
    W: Vec<f64>, // I x K matrix
    a_dist: &'u AD,
    W_dist: &'u WD,
}

impl<AD: Distribution<f64>, WD: Distribution<f64>> LinUtility<'_, AD, WD> {
    pub fn new<'u>(K: usize, a_dist: &'u AD, W_dist: &'u WD) -> LinUtility<'u, AD, WD> {
        return LinUtility::<'u, AD, WD> {
            I: 0,
            K: K,
            a: vec![],
            W: vec![],
            a_dist: a_dist,
            W_dist: W_dist,
        };
    }
}

impl<AD: Distribution<f64>, WD: Distribution<f64>> Utility for LinUtility<'_, AD, WD> {
    fn sample(&mut self, I: usize) -> Option<(usize, f64)> {
        self.I = I;
        self.a = linalg::zeros(I, 1);
        self.W = linalg::zeros(I, self.K);

        // actually sample a, W here... really we should return the whole
        // vectors/matrix, as this is too simple.
        for i in 0..I {
            self.a[i] = self.a_dist.sample(&mut rand::thread_rng());
            for k in 0..self.K {
                self.W[I * k + i] = self.W_dist.sample(&mut rand::thread_rng());
            }
        }

        return None; // inf_f64, inf_f64;
    }

    // fn resample(&mut self, I: usize) -> ... truncate or append samples

    fn values(&self, J: usize, X: &[f64], V: &mut [f64]) {
        // V = W @ X
        for j in 0..J {
            for i in 0..self.I {
                V[self.I * j + i] = 0.0; // needed if we are accumulating...
                for k in 0..self.K {
                    V[self.I * j + i] += self.W[self.I * k + i] * X[self.K * j + k];
                }
            }
        }
    }

    fn eval_UpD0(&self, p: &[f64], U: &mut [f64]) {
        // assert U is self.I x p.len() ? or just panic?
        for i in 0..self.I {
            for j in 0..p.len() {
                U[self.I * j + i] = -self.a[i] * p[j];
            }
        }
    }

    fn eval_UpD1(&self, p: &[f64], U: &mut [f64], DpU: &mut [f64]) {
        // assert U is self.I x p.len() ? or just panic?
        for i in 0..self.I {
            for j in 0..p.len() {
                U[self.I * j + i] = -self.a[i] * p[j];
                DpU[self.I * j + i] = -self.a[i];
            }
        }
    }

    fn eval_UpD2(&self, p: &[f64], U: &mut [f64], DpU: &mut [f64], DppU: &mut [f64]) {
        // assert U is self.I x p.len() ? or just panic?
        for i in 0..self.I {
            for j in 0..p.len() {
                U[self.I * j + i] = -self.a[i] * p[j];
                DpU[self.I * j + i] = -self.a[i];
                DppU[self.I * j + i] = 0.0_f64;
            }
        }
    }
}

pub struct LORUUtility<'u, AD: Distribution<f64>, BD: Distribution<f64>, WD: Distribution<f64>> {
    I: usize,
    K: usize,
    a: Vec<f64>, // I vector
    b: Vec<f64>, // I vector
    W: Vec<f64>, // I x K matrix
    a_dist: &'u AD,
    b_dist: &'u BD,
    W_dist: &'u WD,
}

impl<AD: Distribution<f64>, BD: Distribution<f64>, WD: Distribution<f64>> LORUUtility<'_, AD, BD, WD> {
    pub fn new<'u>(K: usize, a_dist: &'u AD, b_dist: &'u BD, W_dist: &'u WD) -> LORUUtility<'u, AD, BD, WD> {
        return LORUUtility::<'u, AD, BD, WD> {
            I: 0,
            K: K,
            a: vec![],
            b: vec![],
            W: vec![],
            a_dist: a_dist,
            b_dist: b_dist,
            W_dist: W_dist,
        };
    }
}

impl<AD: Distribution<f64>, BD: Distribution<f64>, WD: Distribution<f64>> Utility for LORUUtility<'_, AD, BD, WD> {
    fn sample(&mut self, I: usize) -> Option<(usize, f64)> {
        self.I = I;
        self.a = linalg::zeros(I, 1);
        self.b = linalg::zeros(I, 1);
        self.W = linalg::zeros(I, self.K);

        let mut bimax: usize = 0;
        let mut bmax: f64 = 0.0;

        // actually sample a, W here... really we should return the whole
        // vectors/matrix, as this is too simple.
        for i in 0..I {
            self.a[i] = self.a_dist.sample(&mut rand::thread_rng());
            self.b[i] = self.b_dist.sample(&mut rand::thread_rng());
            if self.b[i] > bmax {
                bimax = i;
                bmax = self.b[i];
            }
            for k in 0..self.K {
                self.W[I * k + i] = self.W_dist.sample(&mut rand::thread_rng());
            }
        }

        return Some((bimax, bmax));
    }

    // fn resample(&mut self, I: usize) -> ... truncate or append samples

    fn values(&self, J: usize, X: &[f64], V: &mut [f64]) {
        // V = W @ X
        for j in 0..J {
            for i in 0..self.I {
                V[self.I * j + i] = 0.0; // needed if we are accumulating...
                for k in 0..self.K {
                    V[self.I * j + i] += self.W[self.I * k + i] * X[self.K * j + k];
                }
            }
        }
    }

    fn eval_UpD0(&self, p: &[f64], U: &mut [f64]) {
        // assert U is self.I x p.len() ? or just panic?
        let mut ri: f64;
        let mut idx: usize;
        for i in 0..self.I {
            for j in 0..p.len() {
                ri = self.b[i] - p[j];
                idx = self.I * j + i;
                if ri > 0.0 {
                    U[idx] = self.a[i] * ri.ln();
                } else {
                    U[idx] = -1.0e20;
                }
            }
        }
    }

    fn eval_UpD1(&self, p: &[f64], U: &mut [f64], DpU: &mut [f64]) {
        // assert U is self.I x p.len() ? or just panic?
        let mut ri: f64;
        let mut idx: usize;
        for i in 0..self.I {
            for j in 0..p.len() {
                ri = self.b[i] - p[j];
                idx = self.I * j + i;
                if ri > 0.0 {
                    U[idx] = self.a[i] * ri.ln();
                    DpU[idx] = -self.a[i] / ri;
                } else {
                    U[idx] = -1.0e20;
                    DpU[idx] = 0.0;
                }
            }
        }
    }

    fn eval_UpD2(&self, p: &[f64], U: &mut [f64], DpU: &mut [f64], DppU: &mut [f64]) {
        // assert U is self.I x p.len() ? or just panic?
        let mut ri: f64;
        let mut idx: usize;
        for i in 0..self.I {
            for j in 0..p.len() {
                ri = self.b[i] - p[j];
                idx = self.I * j + i;
                if ri > 0.0 {
                    U[idx] = self.a[i] * ri.ln();
                    DpU[idx] = -self.a[i] / ri;
                    DppU[idx] = DpU[idx] / ri;
                } else {
                    U[idx] = -1.0e20;
                    DpU[idx] = 0.0;
                    DppU[idx] = -1.0 / self.a[i];
                }
            }
        }
    }
}

pub struct RORUUtility {}

#[cfg(test)]
mod tests {

    use super::*;
    use rand_distr::{LogNormal, Normal};

    #[test]
    fn test_lin_u() {
        let I: usize = 2;
        let J: usize = 10;
        let K: usize = 10;

        let p = linalg::randmat(J, 1, 0.0, 1.0);
        let X = linalg::randmat(K, J, -1.0, 1.0);

        let a_dist = LogNormal::<f64>::new(2.0, 3.0).unwrap();
        let W_dist = Normal::<f64>::new(0.0, 1.0).unwrap();

        let mut V = linalg::zeros(I, J);
        let mut U = linalg::zeros(I, J);
        let mut DpU = linalg::zeros(I, J);
        let mut DppU = linalg::zeros(I, J);

        let mut lu = LinUtility::<LogNormal<f64>, Normal<f64>>::new(K, &a_dist, &W_dist);

        match lu.sample(I) {
            Some(_) => assert!(false),
            None => (),
        };

        for i in 0..I {
            for j in 0..J {
                assert!(V[I * j + i] == 0.0);
            }
        }
        lu.values(J, &X, &mut V); // compute values from samples (W @ X)
        for i in 0..I {
            for j in 0..J {
                assert!(V[I * j + i] != 0.0);
            }
        }

        for i in 0..I {
            for j in 0..J {
                assert!(U[I * j + i] == 0.0);
            }
        }
        lu.eval_UpD0(&p, &mut U);
        for i in 0..I {
            for j in 0..J {
                assert!(U[I * j + i] < 0.0);
            }
        }

        for i in 0..I {
            for j in 0..J {
                assert!(DpU[I * j + i] == 0.0);
            }
        }
        lu.eval_UpD1(&p, &mut U, &mut DpU);
        for i in 0..I {
            for j in 0..J {
                assert!(DpU[I * j + i] != 0.0);
            }
        }

        for i in 0..I {
            for j in 0..J {
                assert!(DppU[I * j + i] == 0.0);
            }
        }
        lu.eval_UpD2(&p, &mut U, &mut DpU, &mut DppU);
        for i in 0..I {
            for j in 0..J {
                assert!(DppU[I * j + i] == 0.0);
            }
        }
    }
}
