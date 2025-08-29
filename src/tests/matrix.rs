use crate::math::matrix::Matrix;

#[test]
fn test_transpose() {
    let data = vec![
        1, 2,
        3, 4,
        5, 6];

    let matrix = Matrix::from_vec(3, 2, data);
    let res = matrix.transpose();

    assert_eq!(res.rows, matrix.cols);
    assert_eq!(res.cols, matrix.rows);
    assert_eq!(res.data, vec![
        1, 3, 5,
        2, 4, 6]
    );
}

#[test]
fn test_gemm_square() {
    let a_data = vec![1, 2, 3, 4];
    let b_data = vec![4, 3, 2, 1];

    let a = Matrix::from_vec(2, 2, a_data);
    let b = Matrix::from_vec(2, 2, b_data);
    let c = a * b;

    assert_eq!(c.rows, 2);
    assert_eq!(c.cols, 2);
    assert_eq!(c.data, vec![8, 5, 20, 13]);
}

#[test]
fn test_gemm_non_square() {
    let a_data = vec![1, 2,
                      3, 4,
                      5, 6];

    let b_data = vec![6, 5, 4,
                      3, 2, 1];

    let a = Matrix::from_vec(3, 2, a_data);
    let b = Matrix::from_vec(2, 3, b_data);

    let c = a * b;

    assert_eq!(c.rows, 3);
    assert_eq!(c.cols, 3);
    assert_eq!(c.data, vec![12, 9, 6,
                            30, 23, 16,
                            48, 37, 26]);
}