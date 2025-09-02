use rayon::iter::ParallelIterator;
use std::cmp::max;
use std::ops::{Add, AddAssign, Mul, Sub};

#[derive(Clone, Debug)]
pub struct Matrix<T>
where
    T: Default + Clone + Copy
    // + Send + Sync
    + Mul<Output = T> + Add<Output = T> + Sub<Output = T>
    + AddAssign,
{
    pub data: Vec<T>,
    pub rows: usize,
    pub cols: usize,
}

impl<T> Matrix<T>
where
    T: Default + Clone + Copy
    // + Send + Sync
    + Mul<Output = T> + Add<Output = T> + Sub<Output = T>
    + AddAssign,
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

    #[inline(always)]
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    #[inline(always)]
    pub fn is_square_power_of_two(&self) -> bool {
        self.is_square() && self.rows.is_power_of_two()
    }

    /// get the value in the matrix at position (`row`, `col`).
    ///
    /// returns none of the specified position is out of bounds.
    #[inline(always)]
    fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.rows && col < self.cols {
            return Some(&self.data[row * self.cols + col])
        }
        None
    }

    /// set the value in the matrix at position (`row`, `col`)
    #[inline(always)]
    fn set(&mut self, row: usize, col: usize, value: T) {
        if row < self.rows && col < self.cols {
            self.data[row * self.cols + col] = value;
        }
    }

    /// transpose a `Matrix<T>`
    pub fn transpose(&self) -> Matrix<T> {
        let new_rows = self.cols;
        let new_cols = self.rows;
        let mut res = Matrix::from_vec(
            new_rows,
            new_cols,
            vec![T::default(); new_rows * new_cols]
        );

        for i in 0..self.cols {
            for j in 0..self.rows {
                res.set(i, j, *self.get(j, i).unwrap());
            }
        }

        res
    }

    ///pad a matrix to ensure that it is a square of a specified size
    /// # Examples
    ///
    /// ```
    /// # #![allow(noop_method_call)]
    /// use flywheel::math::matrix::Matrix;
    ///
    /// let a_data = vec![1, 2,
    ///                   3, 4];
    ///
    /// let a = Matrix::from_vec(2, 2, a_data);
    /// let pad_a = a.pad_to_size(2);
    /// assert_eq!(pad_a.data, vec![1, 2,
    ///                             3, 4]);
    ///
    /// let b_data = vec![1, 2, 3,
    ///                   4, 5, 6];
    ///
    /// let b = Matrix::from_vec(2, 3, b_data);
    /// let pad_b = b.pad_to_size(4);
    /// assert_eq!(pad_b.data, vec![1, 2, 3, 0,
    ///                             4, 5, 6, 0,
    ///                             0, 0, 0, 0,
    ///                             0, 0, 0, 0]);
    /// ```
    #[inline(always)]
    pub fn pad_to_size(&self, size: usize) -> Matrix<T> {
        if self.rows == size && self.cols == size {
            return self.clone();
        }

        let mut result = Matrix::new(size, size);

        // Copy original data
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, *self.get(i, j).unwrap());
            }
        }

        result
    }

    /// breaks a matrix up into quarters.
    /// this expects that the given matrix is square,
    /// and has dimensions which are powers of two.
    ///
    /// to ensure this property it may be worth using [`Matrix::pad_to_size`]
    /// # Examples
    ///
    /// ```
    /// # #![allow(noop_method_call)]
    /// use flywheel::math::matrix::Matrix;
    ///
    /// let a_data = vec![1, 2, 3, 4,
    ///                   5, 6, 7, 8,
    ///                   9, 8, 7, 6,
    ///                   5, 4, 3, 2];
    ///
    /// let a = Matrix::from_vec(4, 4, a_data);
    /// let (a11, a12, a21, a22) = a.quarters();
    ///
    /// assert_eq!(a11.data, vec![1, 2,
    ///                           5, 6]);
    ///
    /// assert_eq!(a12.data, vec![3, 4,
    ///                           7, 8]);
    ///
    /// assert_eq!(a21.data, vec![9, 8,
    ///                           5, 4]);
    ///
    /// assert_eq!(a22.data, vec![7, 6,
    ///                           3, 2]);
    ///
    /// ```
    pub fn quarters(&self) -> (Matrix<T>, Matrix<T>, Matrix<T>, Matrix<T>) {
        assert!(self.is_square_power_of_two());
        let q_rows = self.rows / 2;
        let q_cols = self.cols / 2;

        // first quarter
        let mut data = Vec::with_capacity(q_rows * q_cols);

        for i in 0..q_rows {
            for j in 0..q_cols {
                data.push(*self.get(i, j).unwrap());
            }
        }

        let q1 = Matrix::from_vec(q_rows, q_cols, data);

        // second quarter
        data = Vec::with_capacity(q_rows * q_cols);

        for i in 0..q_rows {
            for j in q_cols..self.cols {
                data.push(*self.get(i, j).unwrap());
            }
        }

        let q2 = Matrix::from_vec(q_rows, q_cols, data);

        // third quarter
        data = Vec::with_capacity(q_rows * q_cols);

        for i in q_rows..self.rows {
            for j in 0..q_cols {
                data.push(*self.get(i, j).unwrap());
            }
        }

        let q3 = Matrix::from_vec(q_rows, q_cols, data);

        // fourth quarter
        data = Vec::with_capacity(q_rows * q_cols);

        for i in q_rows..self.rows {
            for j in q_cols..self.cols {
                data.push(*self.get(i, j).unwrap());
            }
        }

        let q4 = Matrix::from_vec(q_rows, q_cols, data);

        (q1, q2, q3, q4)
    }

    /// merges four quarters into a single matrix,
    /// this expects that all quarters are square matrices of equal size.
    /// # Examples
    ///
    /// ```
    /// # #![allow(noop_method_call)]
    /// use flywheel::math::matrix::Matrix;
    ///
    /// let a_data = vec![1, 2, 3, 4,
    ///                   5, 6, 7, 8,
    ///                   9, 8, 7, 6,
    ///                   5, 4, 3, 2];
    ///
    /// let a = Matrix::from_vec(4, 4, a_data);
    /// let (a11, a12, a21, a22) = a.quarters();
    ///
    /// let merged = Matrix::from_quarters(a11,a12,a21,a22);
    ///
    /// assert_eq!(merged.data, a.data);
    /// ```
    pub fn from_quarters(q1: Matrix<T>, q2: Matrix<T>, q3: Matrix<T>, q4: Matrix<T>) -> Matrix<T> {
        assert!(q1.is_square());
        assert!(q2.is_square());
        assert!(q3.is_square());
        assert!(q4.is_square());
        assert!(q1.rows == q2.rows
            && q2.cols == q3.cols
            && q3.rows == q4.rows
        );

        let (rows, cols) = (q1.rows * 2, q1.cols * 2);
        let mut whole = Matrix::from_vec(
            rows, cols,
            vec![T::default(); rows * cols]
        );

        // add first quarter
        for i in 0..q1.rows {
            for j in 0..q1.cols {
                whole.set(i, j, *q1.get(i, j).unwrap());
            }
        }

        // add second quarter
        for i in 0..q2.rows {
            for j in 0..q2.cols {
                whole.set(i, j + q2.cols, *q2.get(i, j).unwrap());
            }
        }

        // add third quarter
        for i in 0..q3.rows {
            for j in 0..q3.cols {
                whole.set(i + q3.rows, j, *q3.get(i, j).unwrap());
            }
        }

        // add fourth quarter
        for i in 0..q4.rows {
            for j in 0..q4.cols {
                whole.set(i + q4.rows, j + q4.cols, *q4.get(i, j).unwrap());
            }
        }

        whole
    }

    /// multiplies two matrices by the standard
    /// General Matrix-Matrix Multiplication algorithm.
    pub fn gemm(&self, rhs: Matrix<T>) -> Matrix<T> {
        assert_eq!(self.cols, rhs.rows);

        let mut res = Matrix::from_vec(
            self.rows,
            rhs.cols,
            vec![T::default(); self.rows * rhs.cols],
        );

        for i in 0..self.rows {
            for k in 0..self.cols {
                for j in 0..rhs.cols {
                    let aik = self.data[i * self.cols + k];
                    let bkj = rhs.data[k * rhs.cols + j];
                    res.data[i * res.cols + j] += aik * bkj;
                }
            }
        }
        res
    }

    /// multiplies two matrices by Strassen's algorithm
    pub fn strassen(&self, rhs: Matrix<T>, size: usize) -> Matrix<T> {
        let pad_self = self.pad_to_size(size);
        let pad_rhs = rhs.pad_to_size(size);

        assert_eq!(pad_self.rows, pad_rhs.rows);

        if size == 1 {
            return Matrix::from_vec(1, 1, vec![
                *pad_self.get(0, 0).unwrap() * *pad_rhs.get(0, 0).unwrap()
            ])
        }

        let (a11, a12, a21, a22) = pad_self.quarters();
        let (b11, b12, b21, b22) = pad_rhs.quarters();

        let p1 = (a11.clone() + a22.clone()) * (b11.clone() + b22.clone());
        let p2 = (a21.clone() + a22.clone()) * b11.clone();
        let p3 = a11.clone() * (b12.clone() - b22.clone());
        let p4 = a22.clone() * (b21.clone() - b11.clone());
        let p5 = (a11.clone() + a12.clone()) * b22.clone();
        let p6 = (a21 - a11) * (b11 + b12);
        let p7 = (a12 - a22) * (b21 + b22);

        let q0 = p1.clone() + p4.clone() - p5.clone() + p7;
        let q1 = p3.clone() + p5;
        let q2 = p2.clone() + p4;
        let q3 = p1 - p2 + p3 + p6;

        let rows = self.rows;
        let cols = rhs.cols;
        let padded_data = Matrix::from_quarters(q0, q1, q2, q3);
        let mut data = Vec::with_capacity(rows * cols);

        // remove padding
        for i in 0..rows {
            for j in 0..cols {
                data.push(*padded_data.get(i, j).unwrap());
            }
        }

        Matrix::from_vec(rows, cols, data)
    }
}

impl<T> Add for Matrix<T>
where
    T: Default + Clone + Copy
    // + Send + Sync
    + Mul<Output = T> + Add<Output = T> + Sub<Output = T>
    + AddAssign,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows * self.cols {
            result.data[i] = self.data[i] + rhs.data[i];
        }
        result
    }
}

impl<T> Sub for Matrix<T>
where
    T: Default + Clone + Copy
    // + Send + Sync
    + Mul<Output = T> + Add<Output = T> + Sub<Output = T>
    + AddAssign,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows * self.cols {
            result.data[i] = self.data[i] - rhs.data[i];
        }
        result
    }
}

impl<T> Mul for Matrix<T>
where
    T: Default + Clone + Copy
    // + Send + Sync
    + Mul<Output = T> + Add<Output = T> + Sub<Output = T>
    + AddAssign,
{
    type Output = Matrix<T>;

    /// performs general matrix multiplication for two matrices
    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.cols, rhs.rows);

        let largest_dimension = max(
            max(self.rows, self.cols),
            max(rhs.rows, rhs.cols)
        );

        let size = if largest_dimension.is_power_of_two() {
            largest_dimension
        } else {
            2_usize.pow((largest_dimension as f32).log2().ceil() as u32)
        };

        match size {
            0..=256 => self.gemm(rhs),
            _ => self.strassen(rhs, size)
        }
    }
}
