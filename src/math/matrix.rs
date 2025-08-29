use std::ops::{Add, Mul};

#[derive(Clone, Debug)]
pub struct Matrix<T>
where
    T: Default + Clone + Copy + Mul<Output = T> + Add<Output = T>,
{
    pub data: Vec<T>,
    pub rows: usize,
    pub cols: usize,
}

impl<T> Matrix<T>
where
    T: Default + Clone + Copy + Mul<Output = T> + Add<Output = T>,
{
    /// construct a new `Matrix<T>`
    /// with rows and columns specified by `rows` and `cols`,
    /// where each value is the default value of `T`.
    pub fn new(rows: usize, cols: usize) -> Matrix<T> {
        Matrix {
            data: vec![T::default(); rows * cols],
            rows,
            cols,
        }
    }

    /// construct a new `Matrix<T>`
    /// with rows and columns specified by `rows` and `cols`,
    /// and data specified by `data`.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Matrix<T> {
        assert_eq!(rows * cols, data.len());
        Matrix { data, rows, cols }
    }

    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// get the value in the matrix at position (`row`, `col`).
    ///
    /// returns none of the specified position is out of bounds.
    fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.rows && col < self.cols {
            return Some(&self.data[row * self.cols + col])
        }
        println!(
            "({}, {}) is out of bounds ({} {})",
            row, col, self.rows, self.cols
        );
        None
    }

    /// transpose a `Matrix<T>`
    pub fn transpose(&self) -> Matrix<T> {
        let new_rows = self.cols;
        let new_cols = self.rows;
        let mut new_data = Vec::with_capacity(new_rows * new_cols);

        for i in 0..self.cols {
            for j in 0..self.rows {
                new_data.push(*self.get(j, i).unwrap());
            }
        }

        Matrix::from_vec(new_rows, new_cols, new_data)
    }
}

impl<T> Mul for Matrix<T>
where
    T: Default + Clone + Copy + Mul<Output = T> + Add<Output = T>,
{
    type Output = Matrix<T>;

    /// performs general matrix multiplication for two matrices
    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.cols, rhs.rows);

        // todo : for 2x2 matrices, optimize with a direct calculation

        let mut data = Vec::with_capacity(self.rows * rhs.cols);
        let rhs_trans = rhs.transpose();

        let i_bound = self.rows;
        let j_bound = rhs.cols;
        let k_bound = self.cols;

        for i in 0..i_bound {
            for j in 0..j_bound {

                let mut sum = T::default();
                for k in 0..k_bound {
                    let a = self.get(i, k).unwrap().clone();
                    let b = rhs_trans.get(j ,k).unwrap().clone();
                    sum = sum + (a * b);
                }
                data.push(sum);

            }
        }

        Matrix { data, rows: self.rows, cols: rhs.cols }
    }
}
