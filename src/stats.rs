#![allow(non_snake_case)]
#![allow(dead_code)]

use std::fmt;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct FPISolveStats {
    stats: Vec<FPIIterStat>,
    solve_start: Instant,
    iter_start: Instant,
    current_iter: usize,
    solve_duration: Duration,
}

impl FPISolveStats {
    pub fn start(max_iters: usize) -> FPISolveStats {
        return FPISolveStats {
            stats: Vec::<FPIIterStat>::with_capacity(max_iters),
            solve_start: Instant::now(),
            iter_start: Instant::now(),
            current_iter: 0,
            solve_duration: Duration::new(0_u64, 0_u32),
        };
    }

    pub fn finish(&mut self) {
        // TODO: any updates/closure?
    }

    pub fn latest(&self) -> &FPIIterStat {
        return &(self.stats[self.stats.len() - 1]);
    }

    pub fn start_iter(&mut self, iter: usize) {
        self.iter_start = Instant::now();
        self.current_iter = iter;
    }

    pub fn finish_iter(&mut self, p: &Vec<f64>, P: &Vec<f64>, fp_norm: f64, cg_norm: f64) {
        assert!(p.len() == P.len());

        let J = p.len();
        let mut min_p: f64 = p[0];
        let mut max_p: f64 = p[0];
        let mut min_P: f64 = P[0];
        let mut max_P: f64 = P[0];
        let mut none_P: f64 = 1.0_f64 - P[0];
        for j in 1..J {
            min_p = min_p.min(p[j]);
            max_p = max_p.max(p[j]);
            min_P = min_P.min(P[j]);
            max_P = max_P.max(P[j]);
            none_P -= P[j];
        }

        let et = self.solve_start.elapsed().as_micros();
        let it = self.iter_start.elapsed().as_micros();
        self.stats.push(FPIIterStat {
            iter: self.current_iter,
            elapsed_time_us: et,
            iter_time_us: it,
            min_p: min_p,
            max_p: max_p,
            min_prob: min_P,
            max_prob: max_P,
            none_prob: none_P,
            fp_norm: fp_norm,
            cg_norm: cg_norm,
        });
    }
}

impl fmt::Display for FPISolveStats {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "iter\ttot_us\titer_us\tmin p\tmax p\tmin P\tmax P\tP0\t||fp||\t||cg||\n",
        )?;
        for stat in self.stats.iter() {
            write!(f, "{}\n", stat)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct FPIIterStat {
    iter: usize,           // the iteration number
    elapsed_time_us: u128, // us in this iter
    iter_time_us: u128,    // us in this iter
    min_p: f64,            // max price
    max_p: f64,            // max price
    min_prob: f64,         // min choice probability
    max_prob: f64,         // max choice probability
    none_prob: f64,        // "outside good"/"no-choice" probability
    fp_norm: f64,
    cg_norm: f64,
}

impl fmt::Display for FPIIterStat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}\t{}\t{}\t{:0.4}\t{:0.4}\t{:0.4}\t{:0.4}\t{:0.4}\t{:0.8}\t{:0.8}",
            self.iter,
            self.elapsed_time_us,
            self.iter_time_us,
            self.min_p,
            self.max_p,
            self.min_prob,
            self.max_prob,
            self.none_prob,
            self.fp_norm,
            self.cg_norm,
        )
    }
}

// // // // // // // // // // // // // // // // // // // // // // // // // // // // // //
//
// Unit tests
//
// // // // // // // // // // // // // // // // // // // // // // // // // // // // // //

#[cfg(test)]
mod tests {

    use super::*;
    // use std::cmp;
    use rand::Rng;

    #[test]
    fn test_stats() {
        let iters: usize = 5;
        let p: Vec<f64> = vec![0.0, 0.1, 0.3];
        let P: Vec<f64> = vec![0.0, 0.1, 0.3];

        let mut s = FPISolveStats::start(iters);

        for i in 0..iters {
            s.start_iter(i);
            s.finish_iter(&p, &P, 0.0_f64, 0.0_f64);
            println!("{}", s.latest());
        }

        s.finish();

        println!("\n{}", s);
    }
}
