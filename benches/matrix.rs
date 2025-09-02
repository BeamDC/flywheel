use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;
use flywheel::math::matrix::Matrix;

fn bench_2x2_matmul(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let a_data = (0..4)
        .map(|_| rng.gen_range(0..u32::MAX))
        .collect::<Vec<u32>>();

    let b_data = (0..4)
        .map(|_| rng.gen_range(0..u32::MAX))
        .collect::<Vec<u32>>();

    let a = Matrix::from_vec(2, 2, a_data);
    let b = Matrix::from_vec(2, 2, b_data);

    c.bench_function("2x2 matrix multiplication", |bench| bench.iter(|| a.clone() * b.clone()));
}

fn bench_3x3_matmul(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let a_data = (0..9)
        .map(|_| rng.gen_range(0..u32::MAX))
        .collect::<Vec<u32>>();

    let b_data = (0..9)
        .map(|_| rng.gen_range(0..u32::MAX))
        .collect::<Vec<u32>>();

    let a = Matrix::from_vec(3, 3, a_data);
    let b = Matrix::from_vec(3, 3, b_data);

    c.bench_function("3x3 matrix multiplication", |bench| bench.iter(|| a.clone() * b.clone()));
}

fn bench_4x4_matmul(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let a_data = (0..16)
        .map(|_| rng.gen_range(0..u32::MAX))
        .collect::<Vec<u32>>();

    let b_data = (0..16)
        .map(|_| rng.gen_range(0..u32::MAX))
        .collect::<Vec<u32>>();

    let a = Matrix::from_vec(4, 4, a_data);
    let b = Matrix::from_vec(4, 4, b_data);

    c.bench_function("4x4 matrix multiplication", |bench| bench.iter(|| a.clone() * b.clone()));
}

criterion_group!(benches, bench_2x2_matmul, bench_3x3_matmul, bench_4x4_matmul);
criterion_main!(benches);