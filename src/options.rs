#![allow(non_snake_case)]
#![allow(dead_code)]

pub struct FPISolveOptions {
    pub tolerance: f64,
    pub max_iter: usize, // too large, but type compatible
    pub start_at_costs: bool,
    pub corrected: bool,
    pub verbose: bool,
    pub stats_every: usize, // set to 0 to ignore
    pub check: bool,
}

impl FPISolveOptions {
    pub fn default() -> FPISolveOptions {
        return FPISolveOptions {
            tolerance: 1.0e-6,
            max_iter: 1000,
            start_at_costs: true,
            corrected: true,
            verbose: false,
            stats_every: 10,
            check: false,
        };
    }
}
