
pub struct FPISolveOptions {
    samples: u32,
    p0: &Vec<f64>,
    tolerance: f64,
    max_iter: u32,
    corrected: bool,
    verbose: bool,
    check: bool,
}

impl FPISolveOptions {
    pub fn defaults() -> FPISolveOptions {
        return FPISolveOptions {
            samples: 0,
            p0: vec![],
            tolerance: 1.0e-6,
            max_iter: 1000,
            corrected: true,
            verbose: false,
            check: false,
        }
    }
}
