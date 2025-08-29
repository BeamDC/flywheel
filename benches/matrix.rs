use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;
use flywheel::math::matrix::Matrix;

fn bench_gemm_square(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let a_data = (0..(1024 * 1024))
        .map(|_| rng.gen_range(0..u32::MAX))
        .collect::<Vec<u32>>();

    let b_data = (0..(1024 * 1024))
        .map(|_| rng.gen_range(0..u32::MAX))
        .collect::<Vec<u32>>();

    let a = Matrix::from_vec(1024, 1024, a_data);
    let b = Matrix::from_vec(1024, 1024, b_data);

    c.bench_function("gemm square", |bench| bench.iter(|| a.clone() * b.clone()));
}

fn bench_gemm_non_square(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let a_data = (0..(512 * 1024))
        .map(|_| rng.gen_range(0..u32::MAX))
        .collect::<Vec<u32>>();

    let b_data = (0..(512 * 1024))
        .map(|_| rng.gen_range(0..u32::MAX))
        .collect::<Vec<u32>>();

    let a = Matrix::from_vec(1024, 512, a_data);
    let b = Matrix::from_vec(512, 1024, b_data);

    c.bench_function("gemm non square", |bench| bench.iter(|| a.clone() * b.clone()));
}

criterion_group!(benches, bench_gemm_square, bench_gemm_non_square);
criterion_main!(benches);