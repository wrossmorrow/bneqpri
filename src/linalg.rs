#![allow(non_snake_case)]
#![allow(dead_code)]

use rand::Rng;

pub fn zeros(M: usize, N: usize) -> Vec<f64> {
    return vec![0.0_f64; M * N];
}

pub fn ones(M: usize, N: usize) -> Vec<f64> {
    return vec![1.0_f64; M * N];
}

pub fn eye(N: usize) -> Vec<f64> {
    let mut A = zeros(N, N);
    for n in 0..N {
        A[N * n + n] = 1.0;
    }
    return A;
}

pub fn randmat(M: usize, N: usize, l: f64, u: f64) -> Vec<f64> {
    let mut A = zeros(M, N);
    for m in 0..M {
        for n in 0..N {
            A[M * n + m] = rand::thread_rng().gen_range(l..u);
        }
    }
    return A;
}

// TODO: we could test these things but also we're probably
// going to use ndarray
