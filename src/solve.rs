
use crate:firm;
use crate:utility;
use crate:options;
use crate:stats;

pub struct FPISolver{

    pub firms: &Vec<Firm>,
    pub utility: &Utility, // "whatever has the traits"
    pub stats: SolveStats,

    // NOTE: can move to market data

    Fi: Vec<(usize, usize)>, // "[)" style indices segmenting (ordered) firm blocks

    F: u32, // number of firms (firms.len())
    J: u32, // firms.iter().map(|f| f.products).sum()
    Jf: Vec<u32>, // F long, Jf[f] = firms[f].products
    c: Vec<f64>, // c.len() == J
    m: Vec<f64>, // markups, m.len() == J
    pr: Vec<f64>, // profits, pr.len() == F
    P: Vec<f64>, // probabilities, P.len() == J
    L: Vec<f64>, // Lambda "matrix" (diagonals), L.len() == J
    G: Vec<f64>, // Gamma matrix, G.len() == J x J
    z: Vec<f64>, // probabilities, P.len() == J
    phi: Vec<f64>, // "phi" map (p - c - z(p)), phi.len() == J
    cg: Vec<f64>, // combined gradient (Lam(p) * phi(p)), cg.len() == J

    // NOTE: can move to model data

    I: u32, // "individuals" (sample size)
    V: Vec<f64>, // V.len() == I x J
    U: Vec<f64>, // U.len() == I x J
    uimax: Vec<f64>, // uimax.len() == I, for expfloat corrections
    bimax: usize, // for budget corrections
    bmax: f64, // for budget corrections
    DpU: Vec::<f64>, // DpU.len() == I x J
    DppU: Vec::<f64>, // DppU.len() == I x J
    DpUPL: Vec::<f64>, // DpUPL.len() == I x J
    PL: Vec::<f64>, // PL.len() == I x J

}

pub struct FPISolverMarketData {
    Fi: Vec<(usize, usize)>, // "[)" style indices segmenting (ordered) firm blocks
    F: u32, // number of firms (firms.len())
    J: u32, // firms.iter().map(|f| f.products).sum()
    Jf: Vec<u32>, // F long, Jf[f] = firms[f].products
    c: Vec<f64>, // c.len() == J
    m: Vec<f64>, // markups, m.len() == J
    pr: Vec<f64>, // profits, pr.len() == F
    P: Vec<f64>, // probabilities, P.len() == J
    L: Vec<f64>, // Lambda "matrix" (diagonals), L.len() == J
    G: Vec<f64>, // Gamma matrix, G.len() == J x J
    z: Vec<f64>, // probabilities, P.len() == J
    phi: Vec<f64>, // "phi" map (p - c - z(p)), phi.len() == J
    cg: Vec<f64>, // combined gradient (Lam(p) * phi(p)), cg.len() == J
}

pub struct FPISolverModelData {
    I: u32, // "individuals" (sample size)
    V: Vec<f64>, // V.len() == I x J
    U: Vec<f64>, // U.len() == I x J
    uimax: Vec<f64>, // uimax.len() == I, for expfloat corrections
    bimax: usize, // for budget corrections
    bmax: f64, // for budget corrections
    DpU: Vec::<f64>, // DpU.len() == I x J
    DppU: Vec::<f64>, // DppU.len() == I x J
    DpUPL: Vec::<f64>, // DpUPL.len() == I x J
    PL: Vec::<f64>, // PL.len() == I x J
}


impl FPISolver {

    pub fn new(
        firms: &Vec<Firm>,
        utility: &Utility,
    ) -> FPISolver {

        let F = firms.len() as u32;
        let J = firms.iter().map(|f| f.products).sum() as u32;

        let mut solver = FPISolver {
            firms: firms,
            utility: utility,
            stats: FPISolveStats {},
            F: F,
            J: J,
            Fi: Vec::<(usize, usize)>::with_capacity(F),
            Jf: Vec::<usize>::with_capacity(F),
            c: Vec::<f64>::with_capacity(J), 
            V: Vec::<f64>::with_capacity(J), // J x K?

            m: Vec::<f64>::with_capacity(J), // zeros?
            pr: Vec::<f64>::with_capacity(F), // zeros?
            P: Vec::<f64>::with_capacity(J), // zeros?
            L: Vec::<f64>::with_capacity(J), // zeros?
            G: Vec::<f64>::with_capacity(J*J), // zeros?
            z: Vec::<f64>::with_capacity(J), // zeros?
            phi: Vec::<f64>::with_capacity(J), // zeros?
            cg: Vec::<f64>::with_capacity(J), // zeros?

            // NOTE/TODO: can't assign the below until sample size known

            I: I, // "individuals"; 
            U: Vec::<f64>::with_capacity(I*J), // zeros?
            uimax: Vec::<f64>::with_capacity(I), // for expfloat corrections
            bimax: 0_usize, // for budget corrections
            bmax: 0.0_f64, // for budget corrections
            DpU: Vec::<f64>::with_capacity(I*J), // zeros?
            DpUPL: Vec::<f64>::with_capacity(I*J), // zeros?
            PL: Vec::<f64>::with_capacity(I*J), // zeros?
        }

        let mut s: usize = 0;
        for f in firms.iter() {
            let Jf = f.products;
            let e = s + Jf;
            solver.Jf.push(Jf);
            solver.Fi.push((s, e));
            for i in 0..Jf {
                c.push(f.costs[i]);
                // TODO: V; but maybe multi-dimensional?
            }
            s = e;
        }

        return solver;
    }

    pub fn solve(&self, opts: &FPISolveOptions) {

        // # for (annoying but maybe usefuul) consistency with math notation
        // I = samples  # noqa: E741
        // but also, 

        // # random prices in [ c/2 , 3/2c ] if not specified
        // p = (
        //     initial_prices
        //     if initial_prices is not None
        //     else (self.c / 2.0 + 2.0 * np.random.random(self.J))
        // )

        // # sample parameters needed to compute
        // self.utility.sample(I)
        // self.PL = np.zeros((I, self.J))  # logit probabilities (idiosyncratic)
        // self.DpUPL = np.zeros((I, self.J))  # for Lambda/Gamma compute

        // # define reference to corrected or uncorrected step
        // _corrected = corrected and (self.utility.b is not None)
        // step = self.zeta_c if _corrected else self.zeta_u

        // self.max_iter = max_iter
        // self.nrms = np.zeros((max_iter, 2))
        // self.solved = False
        // self.stats = []
        // start = time()
        // for self.iter in range(self.max_iter):

        //     if check:
        //         self.probcheck(p)
        //         self.gradcheck(p)

        //     # compute "step", ie the (corrected?) zeta map
        //     step(p, verbose=verbose)

        //     # test convergence (using step, not combined gradient)
        //     self.nrms[self.iter, 0] = np.max(np.abs(self.phi))
        //     self.nrms[self.iter, 1] = np.max(np.abs(self.L * self.phi))

        //     self.stats.append(
        //         [
        //             self.iter,
        //             time() - start,
        //             p.min(),
        //             p.max(),
        //             self.pr.min(),
        //             self.pr.max(),
        //             self.P.sum(),
        //             self.nrms[self.iter, 0],
        //             self.nrms[self.iter, 1],
        //         ]
        //     )

        //     if verbose:
        //         self.profits(p)
        //         self._progress(p)

        //     if self.nrms[self.iter, 0] <= tolerance:
        //         self.solved = True
        //         break

        //     # fixed-point step, equivalently p -> p - phi = p - ( p - c - z )
        //     p = self.c + self.z

        // self.time = time() - start

        // self.nrms = self.nrms[: self.iter + 1, :]

        // return p

    }

    fn _iterprep(&mut self, p: &[f64]) {
        self._utilities(p);
        self._probabilities(p);
        self._lamgam(p);
        self._markups(p);
    }

    fn _utilities(&mut self, p: &[f64]) {
        // TODO; compute U, uimax?, DpU, and DppU. Sample?

    }

    fn _probabilities(&mut self, p: &[f64]) {

        // uimax[i] = max{ 0, max_j U[i,j] }
        for i in 0..self.I {
            for j in 0..self.J {
                self.uimax[i] = cmp::max(self.uimax[i], self.U[i,j]); // what order?
            }
            self.uimax[i] = cmp::max(0.0_f64, self.uimax[i]);
        }

        // PL[i,j] = exp(U[i,j]-uimax[i])/(exp(-uimax[i]) + sum_k exp(U[i,k]-uimax[i]))
        for i in 0..self.I {
            let mut S = (-uimax[i]).expf(); // e^{-uimax[i]}
            for j in 0..self.J {
                self.PL[i,j] = (self.U[i,j] - uimax[i]).expf(); // bounded above by zero
                S += self.PL[i,j];
            }
            for j in 0..self.J {
                self.PL[i,j] /= S; // divide each by the sum of exp's
            }
        }

        // P = PL' 1 / I
        for j in 0..self.J {
            self.P[j] = 0.0_f64;
            for i in 0..self.I {
                self.P[j] += self.PL[i,j];
            }
            self.P[j] /= (self.I as f64);
        }

    }

    fn _lamgam(&mut self, p: &[f64]) {
        // Compute the "Lambda" and "Gamma" matrices from the papers. Note
        // that "Lambda" (`L`) is a diagonal matrix, but "Gamma" (`G`) is full.
        // 
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
        // a totally reasonable ask.

        let If: f64 = self.I as f64;

        // DpUPL = DpU * PL (componentwise)
        // L = DpUPL' 1 / I
        for j in 0..self.J {
            self.L[j] = 0.0_f64;
            for i in 0..self.I {
                self.DpUPL[i,j] = self.DpU[i,j] * self.PL[i,j];
                self.L[j] += self.DpUPL[i,j];
            }
            self.L[j] /= If;
        }

        // G[Fi[f],Fi[f]] = PL[:,Fi[f]]' DpUPL[:, Fi[f]]
        // 
        // TODO: are we storing as full matrix, or vector of matrices?
        // To the degree the firm-block matrices are all that is required
        // we should probably only store those. 
        let mut s: usize;
        let mut e: usize;
        for f in 0..self.F {
            s = Fi[f][0];
            e = Fi[f][1];
            for j in s..e {
                for k in s..e {
                    self.G[j,k] = 0.0_f64;
                    for i in 0..self.I {
                        self.G[j,k] += self.PL[i,j] * self.DpUPL[i,k];
                    }
                    self.G[j,k] /= If;
                }
            }
        }

    }

    fn _markups(&mut self, p: &[f64]) {
        for j in 0..self.J {
            self.m[j] = p[j] - self.c[j];
        }
    }

    fn _profits(&mut self, p: &[f64], prep: bool) {
        if prep {
            self._utilities(p);
            self._probabilities(p);
            self._markups(p);
        }
        for f in 0..self.F {
            self.pr[f] = 0.0_f64;
            for j in self.Fi[f][0]..self.Fi[f][1] {
                self.pr[f] += self.P[j] * self.m[j];
            }
        }
    }

    fn _zeta_b(&mut self) {
        // z <- \tilde{GAMp}' * m - P
        for f in 0..self.F {
            self.pr[f] = 0.0_f64;
            for j in self.Fi[f][0]..self.Fi[f][1] {
                self.pr[f] += self.P[j] * self.m[j]; // for stats only?
                self.z[j] = - self.P[j];
                for k in self.Fi[f][0]..self.Fi[f][1] {
                    self.z[j] += self.G[k,j] * self.m[k];
                }
            }
        }
    }

    fn _zeta_u(&mut self, p: &[f64]) {
        // Uncorrected "zeta map"
        // 
        //      z <- inv(L(p)) * ( \tilde{G}(p)' * m - P )
        // 
        self._iterprep(p);
        self._zeta_b();
        for j in 0..self.J {
            self.z[j] /= self.L[j];
            self.phi[j] = self.m[j] - self.z[j];
        }
    }

    fn _zeta_c(&mut self, p: &[f64]) {
        // Corrected zeta map
        // 
        //      z <- inv(L(p)) * ( \tilde{G}(p)' * m - P )
        // 
        // for all prices < maxinc, "corrected" otherwise. The correction
        // is a bit complicated for notes here, but in the paper. 

        self._iterprep(p);
        self._zeta_b();

        // nominally z <- inv(L) * z, but with corrections
        // for products whose prices are above the population limit
        // on incomes. The correction is
        //
        //     z[j] = omega[maxinci,j] * ( p[j] - maxinc ) + PL[maxinci,{f}]' * m[{f}]
        //
        for f in 0..self.F {

            let mut prFmi = 0.0_f64;
            for j in self.Fi[f][0]..self.Fi[f][1] {
                prFmi += self.PL[self.bimax, j] * self.m[j];
            }

            for j in self.Fi[f][0]..self.Fi[f][1] {
                if p[j] > bmax { 
                    // correction term - price j is too high; is this right?
                    // p - bmax, not bmax - p[j]?
                    self.z[j] = self.DppU[bimax, j] * (p[j] - bmax) + prFmi;
                } else if self.L[j] >= -1.0e-20 {
                    // L[j] <= 0, so L[j] ~ 0.0, i.e. PL ~ 0.0
                    // use a modification of extended map instead of what is calculated above
                    //
                    //      z[j] = PL[maxinci,{f}]' * m[{f}]
                    //
                    // we exclude the "DppU[uimax, j] * ( p[j] - maxinc )" term expecting
                    // p[j] to be at least close to bmax
                    self.z[j] = prFmi;
                } else {
                    self.z[j] /= self.L[j];
                }
            }
        }

        // compute phi = p - c - z also (self.m updated with _iterprep)
        for j in 0..self.J {
            self.phi[j] = self.m[j] - self.z[j]
        }

    }

    fn _combgrad(&mut self, p: &[f64], prep: bool) {
        if prep {
            self._iterprep(p);
        }
        for j in 0..self.J {
            self.cg[j] = self.L[j] * self.m[j] + self.P[j];
        }
        for f in 0..self.F {
            for j in self.Fi[f][0]..self.Fi[f][1] {
                for k in self.Fi[f][0]..self.Fi[f][1] {
                    self.cg[j] -= self.G[k, j] * self.m[k]
                }
            }
        }
    }

    fn _combgradz(&mut self, p: &[f64], prep: bool) {
        if prep {
            self._zeta_c(p); // includes _iterprep
        }
        for j in 0..self.J {
            self.cg[j] = self.L[j] * self.phi[j];
        }
    }

}

// def summary(self) -> None:
//     if self.solved:
//         print(f"Solved in {self.iter}/{self.max_iter} steps, {self.time} seconds")
//         print(f"fixed-point satisfaction |p-c-z| = {self.nrms[-1, 0]}")
//         print(f"final combined gradient norm = {self.nrms[-1, 1]}")
//     else:
//         print(f"Failed to solve in {self.max_iter} steps, {self.time} seconds.")

// def _progress(self, p: NPArrayType) -> None:
//     print(", ".join([str(s) for s in self.stats[-1]]))  # type: ignore

// def probcheck(self, p: NPArrayType) -> None:

//     print("probcheck: ")

//     self.probabilities(p)
//     P = np.array(self.P)  # force a copy, not reference

//     self.lamgam()
//     DP = -np.array(self.G)  # force copy
//     for j in range(self.J):
//         DP[j, j] += self.L[j]

//     df, dh = np.zeros((self.J, self.J)), np.zeros(10)
//     for h in range(10):
//         H = 10 ** (-h)
//         for j in range(self.J):
//             p[j] += H
//             self.probabilities(p)
//             df[:, j] = (self.P - P) / H
//             p[j] -= H
//         dh[h] = np.abs(df - DP).max()
//         print("  %0.8e: %0.2f %0.10f" % (H, np.log10(dh[h]), dh[h]))

// def gradcheck(self, p: NPArrayType) -> None:

//     print("gradcheck: ")

//     pr = np.array(self.profits(p))  # force a copy, not reference
//     cg = self.combgrad(p)
//     print(f"  cg - cz: { np.abs( cg - self.combgradz(p) ).max() }")

//     df, dh = np.zeros(self.J), np.zeros(10)
//     for h in range(10):
//         H = 10 ** (-h)
//         for f in range(self.F):
//             fi = self.Fis[f]
//             for j in self.Fis[f]:
//                 p[j] += H
//                 self.probabilities(p)
//                 prp = self.P[fi].dot(p[fi] - self.c[fi])
//                 df[j] = (prp - pr[f]) / H
//                 p[j] -= H
//         dh[h] = np.abs(df - cg).max()
//         print("  %0.8e: %0.2f %0.10f" % (H, np.log10(dh[h]), dh[h]))