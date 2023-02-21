#![allow(non_snake_case)]
#![allow(dead_code)]

use crate::firm;
use crate::linalg;
use crate::options;
use crate::stats;
use crate::utility;

use rand::Rng;

pub enum DerivativeOrder {
    Zero,
    One,
    Two,
}

pub struct FPISolver<'a> {
    pub firms: &'a Vec<firm::Firm>,
    pub utility: &'a mut dyn utility::Utility, // "whatever has the traits"
    // stats?
    F: usize,                       // number of firms (firms.len())
    Fi: Vec<(usize, usize, usize)>, // "[)" style indices for firm blocks (start, end, size)
    J: usize,                       // firms.iter().map(|f| f.products).sum()
    K: usize,                       // number of characteristics
    X: Vec<f64>,                    // K x J characteristics
    c: Vec<f64>,                    // c.len() == J
    p: Vec<f64>,                    // p.len() == J
    m: Vec<f64>,                    // markups, m.len() == J
    pr: Vec<f64>,                   // profits, pr.len() == F
    P: Vec<f64>,                    // probabilities, P.len() == J

    _id: FPISolverIterData, // data used during solver iterations
}

pub struct FPISolverIterData {
    I: usize,        // "individuals" (sample size)
    V: Vec<f64>,     // V.len() == I x J
    U: Vec<f64>,     // U.len() == I x J
    uimax: Vec<f64>, // uimax.len() == I, for expfloat corrections
    bimax: usize,    // for budget corrections
    bmax: f64,       // for budget corrections
    DpU: Vec<f64>,   // DpU.len() == I x J
    DppU: Vec<f64>,  // DppU.len() == I x J
    PL: Vec<f64>,    // PL.len() == I x J
    DpUPL: Vec<f64>, // DpUPL.len() == I x J
    L: Vec<f64>,     // L.len() == J, Lambda "matrix" (diagonals)
    G: Vec<f64>,     // G.len() == J x J, Gamma matrix NOTE: we only need \tilde{G}...
    z: Vec<f64>,     // P.len() == J, zeta map
    phi: Vec<f64>,   // phi.len() == J, "phi" map (p - c - z(p))
    cg: Vec<f64>,    // cg.len() == J, combined gradient (L(p) * phi(p), componentwise)
}

impl FPISolverIterData {
    fn empty() -> FPISolverIterData {
        return FPISolverIterData {
            I: 0,           // "individuals";
            U: vec![],      // zeros?
            V: vec![],      //
            uimax: vec![],  // for expfloat corrections
            bimax: 0_usize, // for budget corrections
            bmax: 0.0_f64,  // for budget corrections
            DpU: vec![],    // zeros?
            DppU: vec![],   // zeros?
            DpUPL: vec![],  // zeros?
            PL: vec![],     // zeros?
            L: vec![],
            G: vec![], // TODO: store only blocks
            z: vec![],
            phi: vec![],
            cg: vec![],
        };
    }

    fn sized(I: usize, J: usize) -> FPISolverIterData {
        return FPISolverIterData {
            I: I,                       // "individuals";
            U: linalg::zeros(I, J),     // zeros?
            V: linalg::zeros(I, J),     //
            uimax: linalg::zeros(I, 1), // for expfloat corrections
            bimax: 0_usize,             // for budget corrections
            bmax: 0.0_f64,              // for budget corrections
            DpU: linalg::zeros(I, J),   // zeros?
            DppU: linalg::zeros(I, J),  // zeros?
            DpUPL: linalg::zeros(I, J), // zeros?
            PL: linalg::zeros(I, J),    // zeros?
            L: linalg::zeros(J, 1),
            G: linalg::zeros(J, J), // TODO: store only blocks
            z: linalg::zeros(J, 1),
            phi: linalg::zeros(J, 1),
            cg: linalg::zeros(J, 1),
        };
    }
}

impl FPISolver<'_> {
    pub fn new<'a>(
        firms: &'a Vec<firm::Firm>,
        utility: &'a mut dyn utility::Utility, // anything implementing the traits
    ) -> FPISolver<'a> {
        let F = firms.len();
        let J = firms.iter().map(|f| f.Jf).sum();

        assert!(firms[0].X.len() % firms[0].Jf == 0);
        let K = firms[0].X.len() / firms[0].Jf;
        for f in 1..firms.len() {
            assert!(firms[f].X.len() % firms[f].Jf == 0);
            assert!(firms[f].X.len() / firms[f].Jf == K);
        }

        let mut solver = FPISolver {
            firms: firms,
            utility: utility,
            F: F,
            J: J,
            K: K,
            Fi: Vec::<(usize, usize, usize)>::with_capacity(F),
            X: linalg::zeros(K, J),
            c: linalg::zeros(J, 1),
            p: linalg::zeros(J, 1),
            m: linalg::zeros(J, 1),
            pr: linalg::zeros(F, 1),
            P: linalg::zeros(J, 1),
            _id: FPISolverIterData::empty(), // can't size until we know I
        };

        let mut s: usize = 0;
        for firm in firms.iter() {
            let e = s + firm.Jf;
            solver.Fi.push((s, e, firm.Jf));
            // memcpy? faster?
            for j in 0..firm.Jf {
                solver.c[s + j] = firm.c[j];
                for k in 0..solver.K {
                    let Xidx = solver.K * (s + j) + k;
                    let Fidx = solver.K * j + k;
                    solver.X[Xidx] = firm.X[Fidx];
                }
            }
            s = e;
        }

        return solver;
    }

    pub fn solve(
        &mut self,
        samples: usize,
        p0: Option<&Vec<f64>>,
        opts: &options::FPISolveOptions,
    ) -> Result<Option<stats::FPISolveStats>, Option<stats::FPISolveStats>> {
        // define solver data (we can size now, with samples)
        self._id = FPISolverIterData::sized(samples, self.J);

        // random prices in [ c/2 , 3/2c ] if not specified
        match p0 {
            Some(p0) => {
                for j in 0..self.J {
                    self.p[j] = p0[j];
                }
            }
            None => {
                for j in 0..self.J {
                    let r = rand::thread_rng().gen_range(0.5_f64..1.5_f64);
                    self.p[j] = r * self.c[j];
                }
            }
        }

        // sample parameters needed to compute
        let corrected: bool;
        match self.utility.sample(samples) {
            Some((bi, bm)) => {
                self._id.bimax = bi;
                self._id.bmax = bm;
                corrected = opts.corrected;
            }
            None => {
                corrected = false;
            }
        }

        // pre-computable "values" V for U(p) = F(p|a,b) + V
        self.utility.values(self.J, &self.X, &mut self._id.V);

        // run iterations
        let mut stats = stats::FPISolveStats::start(opts.max_iter);
        let mut solved: bool = false;
        let mut do_stats: bool;
        let mut fp_norm: f64;
        let mut cg_norm: f64;
        for iter in 0..opts.max_iter {
            do_stats = (opts.stats_every > 0) && (iter % opts.stats_every == 0);

            if do_stats {
                stats.start_iter(iter);
            }

            self._iterprep();
            if corrected {
                self._zeta_c();
            } else {
                self._zeta_u();
            }
            self._combgradz();

            fp_norm = self._id.phi[0].abs();
            cg_norm = self._id.cg[0].abs();
            for j in 1..self.J {
                fp_norm = fp_norm.max(self._id.phi[j].abs());
                cg_norm = cg_norm.max(self._id.cg[j].abs());
            }

            // TODO: timing?
            if do_stats {
                stats.finish_iter(&self.p, &self.P, fp_norm, cg_norm);
            }

            // if verbose ? (print "progress")

            // both conditions, or just one?
            if fp_norm <= opts.tolerance || cg_norm <= opts.tolerance {
                solved = true;
                break;
            }

            for j in 0..self.J {
                self.p[j] = self.c[j] + self._id.z[j];
            }
        }
        stats.finish();

        if opts.stats_every > 0 {
            if solved {
                Ok(Some(stats))
            } else {
                Err(Some(stats))
            }
        } else {
            if solved {
                Ok(None)
            } else {
                Err(None)
            }
        }
    }

    /// TODO: copy prices? return prices? add to firms data?
    pub fn prices(&self) -> &Vec<f64> {
        return &(self.p);
    }

    /// compute markups, utilities, their derivatives, probabilities, and
    /// the L/G "matrices" we use in iteratively solving.
    fn _iterprep(&mut self) {
        self._markups();
        self._utilities(DerivativeOrder::Two);
        self._probabilities();
        self._lamgam();
    }

    /// m <- p - c
    fn _markups(&mut self) {
        for j in 0..self.J {
            self.m[j] = self.p[j] - self.c[j];
        }
    }

    /// evaluate utilities given passed interface
    fn _utilities(&mut self, ord: DerivativeOrder) {
        // evaluate utilities
        match ord {
            DerivativeOrder::Zero => {
                self.utility.eval_UpD0(&self.p, &mut self._id.U);
            }
            DerivativeOrder::One => {
                self.utility
                    .eval_UpD1(&self.p, &mut self._id.U, &mut self._id.DpU);
            }
            DerivativeOrder::Two => {
                self.utility.eval_UpD2(
                    &self.p,
                    &mut self._id.U,
                    &mut self._id.DpU,
                    &mut self._id.DppU,
                );
            }
        }

        // U <- U + V
        // uimax[i] = max{ 0, max_j U[i,j] }
        for i in 0..self._id.I {
            for j in 0..self.J {
                self._id.U[self._id.I * j + i] += self._id.V[self._id.I * j + i];
                self._id.uimax[i] = self._id.uimax[i].max(self._id.U[self._id.I * j + i]);
                // what order?
            }
            self._id.uimax[i] = self._id.uimax[i].max(0.0_f64);
        }
    }

    /// compute (mixed) Logit probabilities
    fn _probabilities(&mut self) {
        // Compute
        //
        //      PL[i,j] = exp(U[i,j]-uimax[i]) / S[i]
        //
        // where
        //
        //      S[i] = exp(-uimax[i]) + sum_k exp( U[i,k] - uimax[i] )
        //
        for i in 0..self._id.I {
            let mut S = (-self._id.uimax[i]).exp(); // e^{-uimax[i]}
            for j in 0..self.J {
                self._id.PL[self._id.I * j + i] =
                    (self._id.U[self._id.I * j + i] - self._id.uimax[i]).exp(); // bounded above by zero
                S += self._id.PL[self._id.I * j + i];
            }
            for j in 0..self.J {
                self._id.PL[self._id.I * j + i] /= S; // divide each by the sum of exp's
            }
        }

        // P = PL' 1 / I
        for j in 0..self.J {
            self.P[j] = 0.0_f64;
            for i in 0..self._id.I {
                self.P[j] += self._id.PL[self._id.I * j + i];
            }
            self.P[j] /= self._id.I as f64;
        }
    }

    /// Compute the "Lambda" and "Gamma" matrices from the papers. Note
    /// that "Lambda" (`L`) is a diagonal matrix, but "Gamma" (`G`) is full.
    fn _lamgam(&mut self) {
        // Also note some theory:
        //
        //     DpU * PL -> 0 as p -> b
        //
        // so long as
        //
        //       DpPL -> 0 and - DppU/(DpU)^2 is bounded as p -> b
        //
        // as follows from L'Hopital's rule. Moreover, this is sufficient but
        // not necessary, though boundedness of that ratio of derivatives is
        // a totally reasonable ask. We could enforce this here, or via how
        // DppU and DpU are evaluated.

        let If: f64 = self._id.I as f64;

        // DpUPL = DpU * PL (componentwise)
        // L = DpUPL' 1 / I
        for j in 0..self.J {
            self._id.L[j] = 0.0_f64;
            for i in 0..self._id.I {
                self._id.DpUPL[self._id.I * j + i] =
                    self._id.DpU[self._id.I * j + i] * self._id.PL[self._id.I * j + i];
                self._id.L[j] += self._id.DpUPL[self._id.I * j + i];
            }
            self._id.L[j] /= If;
        }

        // G[Fi[f],Fi[f]] = PL[:,Fi[f]]' DpUPL[:, Fi[f]]
        //
        // TODO: are we storing as full matrix, or vector of matrices?
        // To the degree the firm-block matrices are all that is required
        // we should probably only store those.
        for f in 0..self.F {
            for j in self.Fi[f].0..self.Fi[f].1 {
                for k in self.Fi[f].0..self.Fi[f].1 {
                    self._id.G[self.J * k + j] = 0.0_f64; // G[j,k] in column major
                    for i in 0..self._id.I {
                        self._id.G[self.J * k + j] +=
                            self._id.PL[self._id.I * j + i] * self._id.DpUPL[self._id.I * k + i];
                    }
                    self._id.G[self.J * k + j] /= If;
                }
            }
        }
    }

    // Uncorrected "zeta map"
    //
    //      z <- inv(L(p)) * ( \tilde{G}(p)' * m - P )
    //
    fn _zeta_u(&mut self) {
        self._zeta_b();
        for j in 0..self.J {
            self._id.z[j] /= self._id.L[j];
            self._id.phi[j] = self.m[j] - self._id.z[j];
        }
    }

    // Corrected zeta map
    //
    //      z <- inv(L(p)) * ( \tilde{G}(p)' * m - P )
    //
    // for all prices < maxinc, "corrected" otherwise. The correction
    // is a bit complicated for notes here, but in the paper.
    fn _zeta_c(&mut self) {
        self._zeta_b();

        // nominally z <- inv(L) * z, but with corrections
        // for products whose prices are above the population limit
        // on incomes. The correction is
        //
        //     z[j] = omega[maxinci,j] * ( p[j] - maxinc ) + PL[maxinci,{f}]' * m[{f}]
        //
        for f in 0..self.F {
            let mut prFmi = 0.0_f64;
            for j in self.Fi[f].0..self.Fi[f].1 {
                prFmi += self._id.PL[self._id.I * j + self._id.bimax] * self.m[j];
            }

            for j in self.Fi[f].0..self.Fi[f].1 {
                if self.p[j] > self._id.bmax {
                    // correction term - price j is too high; is this right?
                    // p - bmax, not bmax - p[j]?
                    self._id.z[j] = self._id.DppU[self._id.I * j + self._id.bimax]
                        * (self.p[j] - self._id.bmax)
                        + prFmi;
                } else if self._id.L[j] >= -1.0e-20 {
                    // L[j] <= 0, so L[j] ~ 0.0, i.e. PL ~ 0.0
                    // use a modification of extended map instead of what is calculated above
                    //
                    //      z[j] = PL[maxinci,{f}]' * m[{f}]
                    //
                    // we exclude the "DppU[uimax, j] * ( p[j] - maxinc )" term expecting
                    // p[j] to be at least close to bmax
                    self._id.z[j] = prFmi;
                } else {
                    self._id.z[j] /= self._id.L[j];
                }
            }
        }

        // compute phi = p - c - z also (self.m updated with _iterprep)
        for j in 0..self.J {
            self._id.phi[j] = self.m[j] - self._id.z[j]
        }
    }

    /// z <- \tilde{GAMp}' * m - P
    fn _zeta_b(&mut self) {
        for f in 0..self.F {
            for j in self.Fi[f].0..self.Fi[f].1 {
                self._id.z[j] = -self.P[j];
                for k in self.Fi[f].0..self.Fi[f].1 {
                    self._id.z[j] += self._id.G[self.J * j + k] * self.m[k];
                }
            }
        }
    }

    /// cg <- L(p) phi(p) = L(p) ( p - c - z(p) )
    fn _combgradz(&mut self) {
        for j in 0..self.J {
            self._id.cg[j] = self._id.L[j] * self._id.phi[j];
        }
    }

    fn _combgrad(&mut self) {
        for j in 0..self.J {
            self._id.cg[j] = self._id.L[j] * self.m[j] + self.P[j];
        }
        for f in 0..self.F {
            for j in self.Fi[f].0..self.Fi[f].1 {
                for k in self.Fi[f].0..self.Fi[f].1 {
                    self._id.cg[j] -= self._id.G[self.J * j + k] * self.m[k]
                }
            }
        }
    }

    fn _profits(&mut self, prep: bool) {
        if prep {
            self._markups();
            self._utilities(DerivativeOrder::Zero);
            self._probabilities();
            // don't need lamgam, not iterating
        }
        for f in 0..self.F {
            self.pr[f] = 0.0_f64;
            for j in self.Fi[f].0..self.Fi[f].1 {
                self.pr[f] += self.P[j] * self.m[j];
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use rand_distr::{LogNormal, Normal};
    use rand::Rng;

    #[test]
    fn test_solver() {

        let I = 10;
        let F = 3;
        let K = 5;

        let mut firms: Vec<firm::Firm> = vec![];
        for f in 0..F {
            let Jf: usize = 5; // better to be like random from 3 to 5 or something
            let mut firm = firm::Firm {
                name: "test".to_string(),
                Jf: Jf,
                c: linalg::randmat(Jf, 1, 1.0, 2.0),
                p: linalg::randmat(Jf, 1, 1.5, 2.5),
                X: linalg::randmat(K, Jf, -1.0, 1.0),
            };
            firms.push(firm);
        }

        let a_dist = LogNormal::<f64>::new(2.0, 3.0).unwrap();
        let W_dist = Normal::<f64>::new(0.0, 1.0).unwrap();
        let mut linu = utility::LinUtility::<LogNormal<f64>, Normal<f64>>::new(K, &a_dist, &W_dist);

        let mut solver = FPISolver::new(&firms, &mut linu);

        let opts = options::FPISolveOptions::default();
        match solver.solve(I, None, &opts) {
            Ok(stats) => println!("{}", stats.unwrap().latest()),
            Err(stats) => assert!(false),
        }

    }

}

