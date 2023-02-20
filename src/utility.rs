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
    fn sample(&mut self, I: u32) -> (usize, f64); // return bimax, bmax || inf, inf

    /// Compute the pre-computable, non-price portion of utility
    fn values(&self, J: u32, X: &Vec<f64>, V: &mut Vec<f64>);

    /// Evaluate price component of utility with no derivatives
    fn eval_UpD0(&self, p: &Vec<f64>, &mut U: Vec<f64>);

    /// Evaluate price component of utility with first derivatives
    fn eval_UpD1(&self, p: &Vec<f64>, &mut U: Vec<f64>, &mut DpU: Vec<f64>);

    /// Evaluate price component of utility with second derivatives
    fn eval_UpD2(&self, p: &Vec<f64>, &mut U: Vec<f64>, &mut DpU: Vec<f64>, &mut DppU: Vec<f64>);
}

/// A linear-in-price utility
pub struct LinUtility {
    I: u32,
    K: u32,
    a: Vec<f64>, // I vector
    // no b (budgets) required
    W: Vec<f64>, // I x K matrix
}

impl LinUtility {
    pub fn new(K: u32) -> LinUtility {
        return LinUtility {
            I: 0,
            K: K,
            a: vec![],
            W: vec![],
        };
    }
}

impl Utility for LinUtility {
    fn sample(&mut self, I: u32) -> (usize, f64) {
        self.I = I;

        // actually sample a, W here... e.g. LogNormal and (Multivariate) Normal

        return inf_f64, inf_f64;
    }

    fn values(&self, J: u32, X: &Vec<f64>, V: &mut Vec<f64>) {
        // V = W @ X
        for j in 0..J {
            for i in 0..self.I {
                for k in 0..self.K {
                    V[i,j] += self.W[i,k] * X[k,j];
                }
            }
        }
    }

    fn eval_UpD0(&self, p: &Vec<f64>, &mut U: Vec<f64>) {
        // assert U is self.I x p.len() ? or just panic?
        for i in 0..self.I {
            for j in 0..p.len() {
                U[i,j] = -a[i] * p[j];
            }
        }
    }

    fn eval_UpD1(&self, p: &Vec<f64>, &mut U: Vec<f64>, &mut DpU: Vec<f64>) {
        // assert U is self.I x p.len() ? or just panic?
        for i in 0..self.I {
            for j in 0..p.len() {
                U[i,j] = -a[i] * p[j];
                DpU[i,j] = -a[i];
            }
        }
    }

    fn eval_UpD2(&self, p: &Vec<f64>, &mut U: Vec<f64>, &mut DpU: Vec<f64>, &mut DppU: Vec<f64>) {
        // assert U is self.I x p.len() ? or just panic?
        for i in 0..self.I {
            for j in 0..p.len() {
                U[i,j] = -a[i] * p[j];
                DpU[i,j] = -a[i];
                DppU[i,j] = 0.0_f64;
            }
        }
    }
}

pub struct LORUUtility {

}

pub struct RORUUtility {

}
