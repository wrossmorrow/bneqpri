#![allow(non_snake_case)]
#![allow(dead_code)]

use crate::linalg;

pub struct Firm {
    pub name: String,
    pub Jf: usize,   // I.e., "Jf"
    pub c: Vec<f64>, // Jf elements
    pub p: Vec<f64>, // Jf elements
    pub X: Vec<f64>, // K x Jf elements
                     // NOTE: K = characteristics.len() / products.len()
}

impl Firm {
    pub fn new(name: String, Jf: usize, K: usize) -> Firm {
        return Firm {
            name: name,
            Jf: Jf,
            c: linalg::zeros(Jf, 1),
            p: linalg::zeros(Jf, 1),
            X: linalg::zeros(K, Jf),
        };
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_firm() {
        let mut firm = Firm::new("test".to_string(), 5, 10);
    }
}
