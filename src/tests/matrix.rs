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
fn test_mul_2x2() {
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
fn test_mul_3x3() {
    let a_data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
    let b_data = vec![9, 8, 7, 6, 5, 4, 3, 2, 1];

    let a = Matrix::from_vec(3, 3, a_data);
    let b = Matrix::from_vec(3, 3, b_data);
    let c = a * b;

    assert_eq!(c.rows, 3);
    assert_eq!(c.cols, 3);
    assert_eq!(c.data, vec![30, 24, 18,
                            84, 69, 54,
                            138, 114, 90]);
}

#[test]
fn test_mul_4x4() {
    let a_data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let b_data = vec![16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

    let a = Matrix::from_vec(4, 4, a_data);
    let b = Matrix::from_vec(4, 4, b_data);
    let c = a * b;

    assert_eq!(c.rows, 4);
    assert_eq!(c.cols, 4);
    assert_eq!(c.data, vec![80, 70, 60, 50,
                            240, 214, 188, 162,
                            400, 358, 316, 274,
                            560, 502, 444, 386]);
}

#[test]
fn test_mul_non_square() {
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

#[test]
fn test_pad() {
    let a_data = vec![1, 2,
                      3, 4];

    let b_data = vec![1, 2, 3,
                      4, 5, 6];

    let a = Matrix::from_vec(2, 2, a_data);
    let b = Matrix::from_vec(2, 3, b_data);

    let pad_a = a.pad_to_size(2);
    let pad_b = b.pad_to_size(4);

    assert_eq!(pad_a.data, vec![1, 2,
                                3, 4]
    );

    assert_eq!(pad_b.data, vec![1, 2, 3, 0,
                                4, 5, 6, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0]
    );
}