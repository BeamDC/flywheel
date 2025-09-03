use std::ops::{Add, AddAssign, Mul, Sub};
use std::simd::{f32x1, f32x16, f32x2, f32x32, f32x4, f32x64, f32x8, f64x1, f64x16, f64x2, f64x32, f64x4, f64x64, f64x8, i16x1, i16x16, i16x2, i16x32, i16x4, i16x64, i16x8, i32x1, i32x16, i32x2, i32x32, i32x4, i32x64, i32x8, i64x1, i64x16, i64x2, i64x32, i64x4, i64x64, i64x8, i8x1, i8x16, i8x2, i8x32, i8x4, i8x64, i8x8, isizex1, isizex16, isizex2, isizex32, isizex4, isizex64, isizex8, u16x1, u16x16, u16x2, u16x32, u16x4, u16x64, u16x8, u32x1, u32x16, u32x2, u32x32, u32x4, u32x64, u32x8, u64x1, u64x16, u64x2, u64x32, u64x4, u64x64, u64x8, u8x1, u8x16, u8x2, u8x32, u8x4, u8x64, u8x8, usizex1, usizex16, usizex2, usizex32, usizex4, usizex64, usizex8};
use crate::math::matrix::Matrix;

/// specialized simd matrix multiplications for matrices of size 2x2, 3x3, and 4x4
pub trait MatrixSimd: Sized + Copy {
    type Simd1: Copy + Mul<Self::Simd1, Output = Self::Simd1>
    + Add<Self::Simd1, Output = Self::Simd1>;

    type Simd2: Copy + Mul<Self::Simd2, Output = Self::Simd2>
    + Add<Self::Simd2, Output = Self::Simd2>;

    type Simd4: Copy + Mul<Self::Simd4, Output = Self::Simd4> // needed
    + Add<Self::Simd4, Output = Self::Simd4>;

    type Simd8: Copy + Mul<Self::Simd8, Output = Self::Simd8> // needed
    + Add<Self::Simd8, Output = Self::Simd8>;

    type Simd16: Copy + Mul<Self::Simd16, Output = Self::Simd16>
    + Add<Self::Simd16, Output = Self::Simd16>;

    type Simd32: Copy + Mul<Self::Simd32, Output = Self::Simd32> // needed
    + Add<Self::Simd32, Output = Self::Simd32>;

    type Simd64: Copy + Mul<Self::Simd64, Output = Self::Simd64> // needed
    + Add<Self::Simd64, Output = Self::Simd64>;

    // for creating simd vectors
    fn simd1_from_arr(arr: [Self; 1]) -> Self::Simd1;
    fn simd2_from_arr(arr: [Self; 2]) -> Self::Simd2;
    fn simd4_from_arr(arr: [Self; 4]) -> Self::Simd4;
    fn simd8_from_arr(arr: [Self; 8]) -> Self::Simd8;
    fn simd16_from_arr(arr: [Self; 16]) -> Self::Simd16;
    fn simd32_from_arr(arr: [Self; 32]) -> Self::Simd32;
    fn simd64_from_arr(arr: [Self; 64]) -> Self::Simd64;

    // for getting data back out of a simd vector
    fn simd1_to_arr(simd: Self::Simd1) -> [Self; 1];
    fn simd2_to_arr(simd: Self::Simd2) -> [Self; 2];
    fn simd4_to_arr(simd: Self::Simd4) -> [Self; 4];
    fn simd8_to_arr(simd: Self::Simd8) -> [Self; 8];
    fn simd16_to_arr(simd: Self::Simd16) -> [Self; 16];
    fn simd32_to_arr(simd: Self::Simd32) -> [Self; 32];
    fn simd64_to_arr(simd: Self::Simd64) -> [Self; 64];

    // multiplication for simd vectors
    fn simd1_mul(a: Self::Simd1, b: Self::Simd1) -> Self::Simd1
    where
        Self::Simd1: Mul<Self::Simd1, Output = Self::Simd1>,
    {
        a * b
    }

    fn simd2_mul(a: Self::Simd2, b: Self::Simd2) -> Self::Simd2
    where
        Self::Simd2: Mul<Self::Simd2, Output = Self::Simd2>,
    {
        a * b
    }

    fn simd4_mul(a: Self::Simd4, b: Self::Simd4) -> Self::Simd4
    where
        Self::Simd4: Mul<Self::Simd4, Output = Self::Simd4>,
    {
        a * b
    }

    fn simd8_mul(a: Self::Simd8, b: Self::Simd8) -> Self::Simd8
    where
        Self::Simd8: Mul<Self::Simd8, Output = Self::Simd8>,
    {
        a * b
    }

    fn simd16_mul(a: Self::Simd16, b: Self::Simd16) -> Self::Simd16
    where
        Self::Simd16: Mul<Self::Simd16, Output = Self::Simd16>,
    {
        a * b
    }

    fn simd32_mul(a: Self::Simd32, b: Self::Simd32) -> Self::Simd32
    where
        Self::Simd32: Mul<Self::Simd32, Output = Self::Simd32>,
    {
        a * b
    }

    fn simd64_mul(a: Self::Simd64, b: Self::Simd64) -> Self::Simd64
    where
        Self::Simd64: Mul<Self::Simd64, Output = Self::Simd64>,
    {
        a * b
    }

    // addition for simd vectors
    fn simd1_add(a: Self::Simd1, b: Self::Simd1) -> Self::Simd1
    where
        Self::Simd1: Add<Self::Simd1, Output = Self::Simd1>,
    {
        a + b
    }

    fn simd2_add(a: Self::Simd2, b: Self::Simd2) -> Self::Simd2
    where
        Self::Simd2: Add<Self::Simd2, Output = Self::Simd2>,
    {
        a + b
    }

    fn simd4_add(a: Self::Simd4, b: Self::Simd4) -> Self::Simd4
    where
        Self::Simd4: Add<Self::Simd4, Output = Self::Simd4>,
    {
        a + b
    }

    fn simd8_add(a: Self::Simd8, b: Self::Simd8) -> Self::Simd8
    where
        Self::Simd8: Add<Self::Simd8, Output = Self::Simd8>,
    {
        a + b
    }

    fn simd16_add(a: Self::Simd16, b: Self::Simd16) -> Self::Simd16
    where
        Self::Simd16: Add<Self::Simd16, Output = Self::Simd16>,
    {
        a + b
    }

    fn simd32_add(a: Self::Simd32, b: Self::Simd32) -> Self::Simd32
    where
        Self::Simd32: Add<Self::Simd32, Output = Self::Simd32>,
    {
        a + b
    }

    fn simd64_add(a: Self::Simd64, b: Self::Simd64) -> Self::Simd64
    where
        Self::Simd64: Add<Self::Simd64, Output = Self::Simd64>,
    {
        a + b
    }


    fn simd_2x2_mul(a: &Matrix<Self>, b: &Matrix<Self>) -> Matrix<Self>
    where
        Self: Default + Clone + Copy
        + Mul<Output = Self> + Add<Output = Self> + Sub<Output = Self>
        + AddAssign,
        <Self as MatrixSimd>::Simd8: Mul, <Self as MatrixSimd>::Simd4: Add
    {
        let a_data = [
            a.data[0], a.data[0], a.data[2], a.data[2],
            a.data[1], a.data[1], a.data[3], a.data[3]];

        let b_data = [
            b.data[0], b.data[1], b.data[0], b.data[1],
            b.data[2], b.data[3], b.data[2], b.data[3]];

        let a_simd = Self::simd8_from_arr(a_data);
        let b_simd = Self::simd8_from_arr(b_data);

        let [p0, p1, p2, p3, p4, p5, p6, p7] = Self::simd8_to_arr(
            Self::simd8_mul(a_simd, b_simd)
        );

        let sum_a_simd = Self::simd4_from_arr([p0, p1, p2, p3]);
        let sum_b_simd = Self::simd4_from_arr([p4, p5, p6, p7]);

        let sum_result = Self::simd4_add(sum_a_simd, sum_b_simd);
        let sums = Self::simd4_to_arr(sum_result);

        Matrix::from_vec(
            2, 2,
            vec![sums[0], sums[1], sums[2], sums[3]]
        )
    }

    fn simd_3x3_mul(a: &Matrix<Self>, b: &Matrix<Self>) -> Matrix<Self>
    where
        Self: Default + Clone + Copy
        + Mul<Output = Self> + Add<Output = Self> + Sub<Output = Self>
        + AddAssign,
    {
        // let [a11, a12, a13, a21, a22, a23, a31, a32, a33] = a.data.as_slice();
        // let [b11, b12, b13, b21, b22, b23, b31, b32, b33] = b.data.as_slice();
        let a_data = [
            a.data[0], a.data[1], a.data[2], a.data[0], a.data[1], a.data[2], a.data[0], a.data[1], a.data[2],
            a.data[3], a.data[4], a.data[5], a.data[3], a.data[4], a.data[5], a.data[3], a.data[4], a.data[5],
            a.data[6], a.data[7], a.data[8], a.data[6], a.data[7], a.data[8], a.data[6], a.data[7], a.data[8],
            // filler vals
            a.data[0], a.data[1], a.data[2], a.data[0], a.data[1]
        ];

        let b_data = [
            b.data[0], b.data[3], b.data[6], b.data[1], b.data[4], b.data[7], b.data[2], b.data[5], b.data[8],
            b.data[0], b.data[3], b.data[6], b.data[1], b.data[4], b.data[7], b.data[2], b.data[5], b.data[8],
            b.data[0], b.data[3], b.data[6], b.data[1], b.data[4], b.data[7], b.data[2], b.data[5], b.data[8],
            // filler vals
            b.data[0], b.data[0], b.data[0], b.data[0], b.data[0]
        ];

        let a_simd = Self::simd32_from_arr(a_data);

        let b_simd = Self::simd32_from_arr(b_data);

        let [
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14,
        p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26,
        // filler results
        _, _, _, _, _ ] = Self::simd32_to_arr(
            Self::simd32_mul(a_simd, b_simd)
        );

        // sum the first pair of products for each item
        let sum_1a_simd = Self::simd16_from_arr([
            p0, p3, p6, p9, p12, p15, p18, p21, p24,
            // filler vals
            p0, p0, p0, p0, p0, p0, p0
        ]);

        let sum_1b_simd = Self::simd16_from_arr([
            p1, p4, p7, p10, p13, p16, p19, p22, p25,
            // filler vals
            p1, p1, p1, p1, p1, p1, p1
        ]);

        let [s1, s2, s3, s4, s5, s6, s7, s8, s9, _, _, _, _, _, _, _] = Self::simd16_to_arr(
            Self::simd16_add(sum_1a_simd, sum_1b_simd)
        );
        // sum the second pair of products
        let sum_2a_simd = Self::simd16_from_arr([
            s1, s2, s3, s4, s5, s6, s7, s8, s9,
            // filler vals
            s1, s1, s1, s1, s1, s1, s1
        ]);

        let sum_2b_simd = Self::simd16_from_arr([
            p2, p5, p8, p11, p14, p17, p20, p23, p26,
            // filler vals
            p2, p2, p2, p2, p2, p2, p2
        ]);

        let [r1, r2, r3, r4, r5, r6, r7, r8, r9, _, _, _, _, _, _, _] =Self::simd16_to_arr(
            Self::simd16_add(sum_2a_simd, sum_2b_simd)
        );


        Matrix::from_vec(
            3, 3,
            vec![r1, r2, r3, r4, r5, r6, r7, r8, r9]
        )
    }

    fn simd_4x4_mul(a: &Matrix<Self>, b: &Matrix<Self>) -> Matrix<Self>
    where
        Self: Default + Clone + Copy
        + Mul<Output = Self> + Add<Output = Self> + Sub<Output = Self>
        + AddAssign,
    {
        let a_data = [
            a.data[0], a.data[1], a.data[2], a.data[3], a.data[0], a.data[1],
            a.data[2], a.data[3], a.data[0], a.data[1], a.data[2], a.data[3],
            a.data[0], a.data[1], a.data[2], a.data[3], a.data[4], a.data[5],
            a.data[6], a.data[7], a.data[4], a.data[5], a.data[6], a.data[7],
            a.data[4], a.data[5], a.data[6], a.data[7], a.data[4], a.data[5],
            a.data[6], a.data[7], a.data[8], a.data[9], a.data[10], a.data[11],
            a.data[8], a.data[9], a.data[10], a.data[11], a.data[8], a.data[9],
            a.data[10], a.data[11], a.data[8], a.data[9], a.data[10],
            a.data[11], a.data[12], a.data[13], a.data[14], a.data[15],
            a.data[12], a.data[13], a.data[14], a.data[15], a.data[12],
            a.data[13], a.data[14], a.data[15], a.data[12], a.data[13],
            a.data[14], a.data[15],
        ];

        let b_data = [
            b.data[0], b.data[4], b.data[8], b.data[12], b.data[1], b.data[5], b.data[9], b.data[13],
            b.data[2], b.data[6], b.data[10], b.data[14], b.data[3], b.data[7], b.data[11], b.data[15],
            b.data[0], b.data[4], b.data[8], b.data[12], b.data[1], b.data[5], b.data[9], b.data[13],
            b.data[2], b.data[6], b.data[10], b.data[14], b.data[3], b.data[7], b.data[11], b.data[15],
            b.data[0], b.data[4], b.data[8], b.data[12], b.data[1], b.data[5], b.data[9], b.data[13],
            b.data[2], b.data[6], b.data[10], b.data[14], b.data[3], b.data[7], b.data[11], b.data[15],
            b.data[0], b.data[4], b.data[8], b.data[12], b.data[1], b.data[5], b.data[9], b.data[13],
            b.data[2], b.data[6], b.data[10], b.data[14], b.data[3], b.data[7], b.data[11], b.data[15]
        ];

        let a_simd = Self::simd64_from_arr(a_data);

        let b_simd = Self::simd64_from_arr(b_data);

        let [p0, p1, p2, p3, p4, p5, p6, p7,
        p8, p9, p10, p11, p12, p13, p14, p15,
        p16, p17, p18, p19, p20, p21, p22, p23,
        p24, p25, p26, p27, p28, p29, p30, p31,
        p32, p33, p34, p35, p36, p37, p38, p39,
        p40, p41, p42, p43, p44, p45, p46, p47,
        p48, p49, p50, p51, p52, p53, p54, p55,
        p56, p57, p58, p59, p60, p61, p62, p63] = Self::simd64_to_arr(
            Self::simd64_mul(a_simd, b_simd)
        );

        // first sum
        let sum_a1_simd = Self::simd32_from_arr([
            p0, p2, p4, p6, p8, p10, p12, p14,
            p16, p18, p20, p22, p24, p26, p28, p30,
            p32, p34, p36, p38, p40, p42, p44, p46,
            p48, p50, p52, p54, p56, p58, p60, p62
        ]);

        let sum_b1_simd = Self::simd32_from_arr([
            p1, p3, p5, p7, p9, p11, p13, p15,
            p17, p19, p21, p23, p25, p27, p29, p31,
            p33, p35, p37, p39, p41, p43, p45, p47,
            p49, p51, p53, p55, p57, p59, p61, p63
        ]);

        let [s0, s1, s2, s3, s4, s5, s6, s7, s8,
        s9, s10, s11, s12, s13, s14, s15,
        s16, s17, s18, s19, s20, s21, s22,
        s23, s24, s25, s26, s27, s28, s29, s30, s31] = Self::simd32_to_arr(
            Self::simd32_add(sum_a1_simd, sum_b1_simd)
        );

        // second sum
        let sum_a2_simd = Self::simd16_from_arr([
            s0, s2, s4, s6, s8, s10, s12, s14, s16,
            s18, s20, s22, s24, s26, s28, s30
        ]);

        let sum_b2_simd = Self::simd16_from_arr([
            s1, s3, s5, s7, s9, s11, s13, s15, s17,
            s19, s21, s23, s25, s27, s29, s31
        ]);

        let [r1,r2,r3,r4,r5,r6,r7,r8,
        r9,r10,r11,r12,r13,r14,r15,r16] = Self::simd16_to_arr(
            Self::simd16_add(sum_a2_simd, sum_b2_simd)
        );

        Matrix::from_vec(
            4, 4,
            vec![r1, r2, r3, r4,
                 r5, r6, r7, r8,
                 r9, r10, r11, r12,
                 r13, r14, r15, r16]
        )
    }
}

impl MatrixSimd for f32 {
    type Simd1 = f32x1;
    type Simd2 = f32x2;
    type Simd4 = f32x4;
    type Simd8 = f32x8;
    type Simd16 = f32x16;
    type Simd32 = f32x32;
    type Simd64 = f32x64;

    fn simd1_from_arr(arr: [Self; 1]) -> Self::Simd1 {
        f32x1::from_array(arr)
    }

    fn simd2_from_arr(arr: [Self; 2]) -> Self::Simd2 {
        f32x2::from_array(arr)
    }

    fn simd4_from_arr(arr: [Self; 4]) -> Self::Simd4 {
        f32x4::from_array(arr)
    }

    fn simd8_from_arr(arr: [Self; 8]) -> Self::Simd8 {
        f32x8::from_array(arr)
    }

    fn simd16_from_arr(arr: [Self; 16]) -> Self::Simd16 {
        f32x16::from_array(arr)
    }

    fn simd32_from_arr(arr: [Self; 32]) -> Self::Simd32 {
        f32x32::from_array(arr)
    }

    fn simd64_from_arr(arr: [Self; 64]) -> Self::Simd64 {
        f32x64::from_array(arr)
    }

    fn simd1_to_arr(simd: Self::Simd1) -> [Self; 1] {
        simd.to_array()
    }

    fn simd2_to_arr(simd: Self::Simd2) -> [Self; 2] {
        simd.to_array()
    }

    fn simd4_to_arr(simd: Self::Simd4) -> [Self; 4] {
        simd.to_array()
    }

    fn simd8_to_arr(simd: Self::Simd8) -> [Self; 8] {
        simd.to_array()
    }

    fn simd16_to_arr(simd: Self::Simd16) -> [Self; 16] {
        simd.to_array()
    }

    fn simd32_to_arr(simd: Self::Simd32) -> [Self; 32] {
        simd.to_array()
    }

    fn simd64_to_arr(simd: Self::Simd64) -> [Self; 64] {
        simd.to_array()
    }
}

impl MatrixSimd for f64 {
    type Simd1 = f64x1;
    type Simd2 = f64x2;
    type Simd4 = f64x4;
    type Simd8 = f64x8;
    type Simd16 = f64x16;
    type Simd32 = f64x32;
    type Simd64 = f64x64;

    fn simd1_from_arr(arr: [Self; 1]) -> Self::Simd1 {
        f64x1::from_array(arr)
    }

    fn simd2_from_arr(arr: [Self; 2]) -> Self::Simd2 {
        f64x2::from_array(arr)
    }

    fn simd4_from_arr(arr: [Self; 4]) -> Self::Simd4 {
        f64x4::from_array(arr)
    }

    fn simd8_from_arr(arr: [Self; 8]) -> Self::Simd8 {
        f64x8::from_array(arr)
    }

    fn simd16_from_arr(arr: [Self; 16]) -> Self::Simd16 {
        f64x16::from_array(arr)
    }

    fn simd32_from_arr(arr: [Self; 32]) -> Self::Simd32 {
        f64x32::from_array(arr)
    }

    fn simd64_from_arr(arr: [Self; 64]) -> Self::Simd64 {
        f64x64::from_array(arr)
    }

    fn simd1_to_arr(simd: Self::Simd1) -> [Self; 1] {
        simd.to_array()
    }

    fn simd2_to_arr(simd: Self::Simd2) -> [Self; 2] {
        simd.to_array()
    }

    fn simd4_to_arr(simd: Self::Simd4) -> [Self; 4] {
        simd.to_array()
    }

    fn simd8_to_arr(simd: Self::Simd8) -> [Self; 8] {
        simd.to_array()
    }

    fn simd16_to_arr(simd: Self::Simd16) -> [Self; 16] {
        simd.to_array()
    }

    fn simd32_to_arr(simd: Self::Simd32) -> [Self; 32] {
        simd.to_array()
    }

    fn simd64_to_arr(simd: Self::Simd64) -> [Self; 64] {
        simd.to_array()
    }
}

impl MatrixSimd for i8 {
    type Simd1 = i8x1;
    type Simd2 = i8x2;
    type Simd4 = i8x4;
    type Simd8 = i8x8;
    type Simd16 = i8x16;
    type Simd32 = i8x32;
    type Simd64 = i8x64;

    fn simd1_from_arr(arr: [Self; 1]) -> Self::Simd1 {
        i8x1::from_array(arr)
    }

    fn simd2_from_arr(arr: [Self; 2]) -> Self::Simd2 {
        i8x2::from_array(arr)
    }

    fn simd4_from_arr(arr: [Self; 4]) -> Self::Simd4 {
        i8x4::from_array(arr)
    }

    fn simd8_from_arr(arr: [Self; 8]) -> Self::Simd8 {
        i8x8::from_array(arr)
    }

    fn simd16_from_arr(arr: [Self; 16]) -> Self::Simd16 {
        i8x16::from_array(arr)
    }

    fn simd32_from_arr(arr: [Self; 32]) -> Self::Simd32 {
        i8x32::from_array(arr)
    }

    fn simd64_from_arr(arr: [Self; 64]) -> Self::Simd64 {
        i8x64::from_array(arr)
    }

    fn simd1_to_arr(simd: Self::Simd1) -> [Self; 1] {
        simd.to_array()
    }

    fn simd2_to_arr(simd: Self::Simd2) -> [Self; 2] {
        simd.to_array()
    }

    fn simd4_to_arr(simd: Self::Simd4) -> [Self; 4] {
        simd.to_array()
    }

    fn simd8_to_arr(simd: Self::Simd8) -> [Self; 8] {
        simd.to_array()
    }

    fn simd16_to_arr(simd: Self::Simd16) -> [Self; 16] {
        simd.to_array()
    }

    fn simd32_to_arr(simd: Self::Simd32) -> [Self; 32] {
        simd.to_array()
    }

    fn simd64_to_arr(simd: Self::Simd64) -> [Self; 64] {
        simd.to_array()
    }
}

impl MatrixSimd for i16 {
    type Simd1 = i16x1;
    type Simd2 = i16x2;
    type Simd4 = i16x4;
    type Simd8 = i16x8;
    type Simd16 = i16x16;
    type Simd32 = i16x32;
    type Simd64 = i16x64;

    fn simd1_from_arr(arr: [Self; 1]) -> Self::Simd1 {
        i16x1::from_array(arr)
    }

    fn simd2_from_arr(arr: [Self; 2]) -> Self::Simd2 {
        i16x2::from_array(arr)
    }

    fn simd4_from_arr(arr: [Self; 4]) -> Self::Simd4 {
        i16x4::from_array(arr)
    }

    fn simd8_from_arr(arr: [Self; 8]) -> Self::Simd8 {
        i16x8::from_array(arr)
    }

    fn simd16_from_arr(arr: [Self; 16]) -> Self::Simd16 {
        i16x16::from_array(arr)
    }

    fn simd32_from_arr(arr: [Self; 32]) -> Self::Simd32 {
        i16x32::from_array(arr)
    }

    fn simd64_from_arr(arr: [Self; 64]) -> Self::Simd64 {
        i16x64::from_array(arr)
    }

    fn simd1_to_arr(simd: Self::Simd1) -> [Self; 1] {
        simd.to_array()
    }

    fn simd2_to_arr(simd: Self::Simd2) -> [Self; 2] {
        simd.to_array()
    }

    fn simd4_to_arr(simd: Self::Simd4) -> [Self; 4] {
        simd.to_array()
    }

    fn simd8_to_arr(simd: Self::Simd8) -> [Self; 8] {
        simd.to_array()
    }

    fn simd16_to_arr(simd: Self::Simd16) -> [Self; 16] {
        simd.to_array()
    }

    fn simd32_to_arr(simd: Self::Simd32) -> [Self; 32] {
        simd.to_array()
    }

    fn simd64_to_arr(simd: Self::Simd64) -> [Self; 64] {
        simd.to_array()
    }
}

impl MatrixSimd for i32 {
    type Simd1 = i32x1;
    type Simd2 = i32x2;
    type Simd4 = i32x4;
    type Simd8 = i32x8;
    type Simd16 = i32x16;
    type Simd32 = i32x32;
    type Simd64 = i32x64;

    fn simd1_from_arr(arr: [Self; 1]) -> Self::Simd1 {
        i32x1::from_array(arr)
    }

    fn simd2_from_arr(arr: [Self; 2]) -> Self::Simd2 {
        i32x2::from_array(arr)
    }

    fn simd4_from_arr(arr: [Self; 4]) -> Self::Simd4 {
        i32x4::from_array(arr)
    }

    fn simd8_from_arr(arr: [Self; 8]) -> Self::Simd8 {
        i32x8::from_array(arr)
    }

    fn simd16_from_arr(arr: [Self; 16]) -> Self::Simd16 {
        i32x16::from_array(arr)
    }

    fn simd32_from_arr(arr: [Self; 32]) -> Self::Simd32 {
        i32x32::from_array(arr)
    }

    fn simd64_from_arr(arr: [Self; 64]) -> Self::Simd64 {
        i32x64::from_array(arr)
    }

    fn simd1_to_arr(simd: Self::Simd1) -> [Self; 1] {
        simd.to_array()
    }

    fn simd2_to_arr(simd: Self::Simd2) -> [Self; 2] {
        simd.to_array()
    }

    fn simd4_to_arr(simd: Self::Simd4) -> [Self; 4] {
        simd.to_array()
    }

    fn simd8_to_arr(simd: Self::Simd8) -> [Self; 8] {
        simd.to_array()
    }

    fn simd16_to_arr(simd: Self::Simd16) -> [Self; 16] {
        simd.to_array()
    }

    fn simd32_to_arr(simd: Self::Simd32) -> [Self; 32] {
        simd.to_array()
    }

    fn simd64_to_arr(simd: Self::Simd64) -> [Self; 64] {
        simd.to_array()
    }
}

impl MatrixSimd for i64 {
    type Simd1 = i64x1;
    type Simd2 = i64x2;
    type Simd4 = i64x4;
    type Simd8 = i64x8;
    type Simd16 = i64x16;
    type Simd32 = i64x32;
    type Simd64 = i64x64;

    fn simd1_from_arr(arr: [Self; 1]) -> Self::Simd1 {
        i64x1::from_array(arr)
    }

    fn simd2_from_arr(arr: [Self; 2]) -> Self::Simd2 {
        i64x2::from_array(arr)
    }

    fn simd4_from_arr(arr: [Self; 4]) -> Self::Simd4 {
        i64x4::from_array(arr)
    }

    fn simd8_from_arr(arr: [Self; 8]) -> Self::Simd8 {
        i64x8::from_array(arr)
    }

    fn simd16_from_arr(arr: [Self; 16]) -> Self::Simd16 {
        i64x16::from_array(arr)
    }

    fn simd32_from_arr(arr: [Self; 32]) -> Self::Simd32 {
        i64x32::from_array(arr)
    }

    fn simd64_from_arr(arr: [Self; 64]) -> Self::Simd64 {
        i64x64::from_array(arr)
    }

    fn simd1_to_arr(simd: Self::Simd1) -> [Self; 1] {
        simd.to_array()
    }

    fn simd2_to_arr(simd: Self::Simd2) -> [Self; 2] {
        simd.to_array()
    }

    fn simd4_to_arr(simd: Self::Simd4) -> [Self; 4] {
        simd.to_array()
    }

    fn simd8_to_arr(simd: Self::Simd8) -> [Self; 8] {
        simd.to_array()
    }

    fn simd16_to_arr(simd: Self::Simd16) -> [Self; 16] {
        simd.to_array()
    }

    fn simd32_to_arr(simd: Self::Simd32) -> [Self; 32] {
        simd.to_array()
    }

    fn simd64_to_arr(simd: Self::Simd64) -> [Self; 64] {
        simd.to_array()
    }
}

impl MatrixSimd for isize {
    type Simd1 = isizex1;
    type Simd2 = isizex2;
    type Simd4 = isizex4;
    type Simd8 = isizex8;
    type Simd16 = isizex16;
    type Simd32 = isizex32;
    type Simd64 = isizex64;

    fn simd1_from_arr(arr: [Self; 1]) -> Self::Simd1 {
        isizex1::from_array(arr)
    }

    fn simd2_from_arr(arr: [Self; 2]) -> Self::Simd2 {
        isizex2::from_array(arr)
    }

    fn simd4_from_arr(arr: [Self; 4]) -> Self::Simd4 {
        isizex4::from_array(arr)
    }

    fn simd8_from_arr(arr: [Self; 8]) -> Self::Simd8 {
        isizex8::from_array(arr)
    }

    fn simd16_from_arr(arr: [Self; 16]) -> Self::Simd16 {
        isizex16::from_array(arr)
    }

    fn simd32_from_arr(arr: [Self; 32]) -> Self::Simd32 {
        isizex32::from_array(arr)
    }

    fn simd64_from_arr(arr: [Self; 64]) -> Self::Simd64 {
        isizex64::from_array(arr)
    }

    fn simd1_to_arr(simd: Self::Simd1) -> [Self; 1] {
        simd.to_array()
    }

    fn simd2_to_arr(simd: Self::Simd2) -> [Self; 2] {
        simd.to_array()
    }

    fn simd4_to_arr(simd: Self::Simd4) -> [Self; 4] {
        simd.to_array()
    }

    fn simd8_to_arr(simd: Self::Simd8) -> [Self; 8] {
        simd.to_array()
    }

    fn simd16_to_arr(simd: Self::Simd16) -> [Self; 16] {
        simd.to_array()
    }

    fn simd32_to_arr(simd: Self::Simd32) -> [Self; 32] {
        simd.to_array()
    }

    fn simd64_to_arr(simd: Self::Simd64) -> [Self; 64] {
        simd.to_array()
    }
}

impl MatrixSimd for u8 {
    type Simd1 = u8x1;
    type Simd2 = u8x2;
    type Simd4 = u8x4;
    type Simd8 = u8x8;
    type Simd16 = u8x16;
    type Simd32 = u8x32;
    type Simd64 = u8x64;

    fn simd1_from_arr(arr: [Self; 1]) -> Self::Simd1 {
        u8x1::from_array(arr)
    }

    fn simd2_from_arr(arr: [Self; 2]) -> Self::Simd2 {
        u8x2::from_array(arr)
    }

    fn simd4_from_arr(arr: [Self; 4]) -> Self::Simd4 {
        u8x4::from_array(arr)
    }

    fn simd8_from_arr(arr: [Self; 8]) -> Self::Simd8 {
        u8x8::from_array(arr)
    }

    fn simd16_from_arr(arr: [Self; 16]) -> Self::Simd16 {
        u8x16::from_array(arr)
    }

    fn simd32_from_arr(arr: [Self; 32]) -> Self::Simd32 {
        u8x32::from_array(arr)
    }

    fn simd64_from_arr(arr: [Self; 64]) -> Self::Simd64 {
        u8x64::from_array(arr)
    }

    fn simd1_to_arr(simd: Self::Simd1) -> [Self; 1] {
        simd.to_array()
    }

    fn simd2_to_arr(simd: Self::Simd2) -> [Self; 2] {
        simd.to_array()
    }

    fn simd4_to_arr(simd: Self::Simd4) -> [Self; 4] {
        simd.to_array()
    }

    fn simd8_to_arr(simd: Self::Simd8) -> [Self; 8] {
        simd.to_array()
    }

    fn simd16_to_arr(simd: Self::Simd16) -> [Self; 16] {
        simd.to_array()
    }

    fn simd32_to_arr(simd: Self::Simd32) -> [Self; 32] {
        simd.to_array()
    }

    fn simd64_to_arr(simd: Self::Simd64) -> [Self; 64] {
        simd.to_array()
    }
}

impl MatrixSimd for u16 {
    type Simd1 = u16x1;
    type Simd2 = u16x2;
    type Simd4 = u16x4;
    type Simd8 = u16x8;
    type Simd16 = u16x16;
    type Simd32 = u16x32;
    type Simd64 = u16x64;

    fn simd1_from_arr(arr: [Self; 1]) -> Self::Simd1 {
        u16x1::from_array(arr)
    }

    fn simd2_from_arr(arr: [Self; 2]) -> Self::Simd2 {
        u16x2::from_array(arr)
    }

    fn simd4_from_arr(arr: [Self; 4]) -> Self::Simd4 {
        u16x4::from_array(arr)
    }

    fn simd8_from_arr(arr: [Self; 8]) -> Self::Simd8 {
        u16x8::from_array(arr)
    }

    fn simd16_from_arr(arr: [Self; 16]) -> Self::Simd16 {
        u16x16::from_array(arr)
    }

    fn simd32_from_arr(arr: [Self; 32]) -> Self::Simd32 {
        u16x32::from_array(arr)
    }

    fn simd64_from_arr(arr: [Self; 64]) -> Self::Simd64 {
        u16x64::from_array(arr)
    }

    fn simd1_to_arr(simd: Self::Simd1) -> [Self; 1] {
        simd.to_array()
    }

    fn simd2_to_arr(simd: Self::Simd2) -> [Self; 2] {
        simd.to_array()
    }

    fn simd4_to_arr(simd: Self::Simd4) -> [Self; 4] {
        simd.to_array()
    }

    fn simd8_to_arr(simd: Self::Simd8) -> [Self; 8] {
        simd.to_array()
    }

    fn simd16_to_arr(simd: Self::Simd16) -> [Self; 16] {
        simd.to_array()
    }

    fn simd32_to_arr(simd: Self::Simd32) -> [Self; 32] {
        simd.to_array()
    }

    fn simd64_to_arr(simd: Self::Simd64) -> [Self; 64] {
        simd.to_array()
    }
}

impl MatrixSimd for u32 {
    type Simd1 = u32x1;
    type Simd2 = u32x2;
    type Simd4 = u32x4;
    type Simd8 = u32x8;
    type Simd16 = u32x16;
    type Simd32 = u32x32;
    type Simd64 = u32x64;

    fn simd1_from_arr(arr: [Self; 1]) -> Self::Simd1 {
        u32x1::from_array(arr)
    }

    fn simd2_from_arr(arr: [Self; 2]) -> Self::Simd2 {
        u32x2::from_array(arr)
    }

    fn simd4_from_arr(arr: [Self; 4]) -> Self::Simd4 {
        u32x4::from_array(arr)
    }

    fn simd8_from_arr(arr: [Self; 8]) -> Self::Simd8 {
        u32x8::from_array(arr)
    }

    fn simd16_from_arr(arr: [Self; 16]) -> Self::Simd16 {
        u32x16::from_array(arr)
    }

    fn simd32_from_arr(arr: [Self; 32]) -> Self::Simd32 {
        u32x32::from_array(arr)
    }

    fn simd64_from_arr(arr: [Self; 64]) -> Self::Simd64 {
        u32x64::from_array(arr)
    }

    fn simd1_to_arr(simd: Self::Simd1) -> [Self; 1] {
        simd.to_array()
    }

    fn simd2_to_arr(simd: Self::Simd2) -> [Self; 2] {
        simd.to_array()
    }

    fn simd4_to_arr(simd: Self::Simd4) -> [Self; 4] {
        simd.to_array()
    }

    fn simd8_to_arr(simd: Self::Simd8) -> [Self; 8] {
        simd.to_array()
    }

    fn simd16_to_arr(simd: Self::Simd16) -> [Self; 16] {
        simd.to_array()
    }

    fn simd32_to_arr(simd: Self::Simd32) -> [Self; 32] {
        simd.to_array()
    }

    fn simd64_to_arr(simd: Self::Simd64) -> [Self; 64] {
        simd.to_array()
    }
}

impl MatrixSimd for u64 {
    type Simd1 = u64x1;
    type Simd2 = u64x2;
    type Simd4 = u64x4;
    type Simd8 = u64x8;
    type Simd16 = u64x16;
    type Simd32 = u64x32;
    type Simd64 = u64x64;

    fn simd1_from_arr(arr: [Self; 1]) -> Self::Simd1 {
        u64x1::from_array(arr)
    }

    fn simd2_from_arr(arr: [Self; 2]) -> Self::Simd2 {
        u64x2::from_array(arr)
    }

    fn simd4_from_arr(arr: [Self; 4]) -> Self::Simd4 {
        u64x4::from_array(arr)
    }

    fn simd8_from_arr(arr: [Self; 8]) -> Self::Simd8 {
        u64x8::from_array(arr)
    }

    fn simd16_from_arr(arr: [Self; 16]) -> Self::Simd16 {
        u64x16::from_array(arr)
    }

    fn simd32_from_arr(arr: [Self; 32]) -> Self::Simd32 {
        u64x32::from_array(arr)
    }

    fn simd64_from_arr(arr: [Self; 64]) -> Self::Simd64 {
        u64x64::from_array(arr)
    }

    fn simd1_to_arr(simd: Self::Simd1) -> [Self; 1] {
        simd.to_array()
    }

    fn simd2_to_arr(simd: Self::Simd2) -> [Self; 2] {
        simd.to_array()
    }

    fn simd4_to_arr(simd: Self::Simd4) -> [Self; 4] {
        simd.to_array()
    }

    fn simd8_to_arr(simd: Self::Simd8) -> [Self; 8] {
        simd.to_array()
    }

    fn simd16_to_arr(simd: Self::Simd16) -> [Self; 16] {
        simd.to_array()
    }

    fn simd32_to_arr(simd: Self::Simd32) -> [Self; 32] {
        simd.to_array()
    }

    fn simd64_to_arr(simd: Self::Simd64) -> [Self; 64] {
        simd.to_array()
    }
}

impl MatrixSimd for usize {
    type Simd1 = usizex1;
    type Simd2 = usizex2;
    type Simd4 = usizex4;
    type Simd8 = usizex8;
    type Simd16 = usizex16;
    type Simd32 = usizex32;
    type Simd64 = usizex64;

    fn simd1_from_arr(arr: [Self; 1]) -> Self::Simd1 {
        usizex1::from_array(arr)
    }

    fn simd2_from_arr(arr: [Self; 2]) -> Self::Simd2 {
        usizex2::from_array(arr)
    }

    fn simd4_from_arr(arr: [Self; 4]) -> Self::Simd4 {
        usizex4::from_array(arr)
    }

    fn simd8_from_arr(arr: [Self; 8]) -> Self::Simd8 {
        usizex8::from_array(arr)
    }

    fn simd16_from_arr(arr: [Self; 16]) -> Self::Simd16 {
        usizex16::from_array(arr)
    }

    fn simd32_from_arr(arr: [Self; 32]) -> Self::Simd32 {
        usizex32::from_array(arr)
    }

    fn simd64_from_arr(arr: [Self; 64]) -> Self::Simd64 {
        usizex64::from_array(arr)
    }

    fn simd1_to_arr(simd: Self::Simd1) -> [Self; 1] {
        simd.to_array()
    }

    fn simd2_to_arr(simd: Self::Simd2) -> [Self; 2] {
        simd.to_array()
    }

    fn simd4_to_arr(simd: Self::Simd4) -> [Self; 4] {
        simd.to_array()
    }

    fn simd8_to_arr(simd: Self::Simd8) -> [Self; 8] {
        simd.to_array()
    }

    fn simd16_to_arr(simd: Self::Simd16) -> [Self; 16] {
        simd.to_array()
    }

    fn simd32_to_arr(simd: Self::Simd32) -> [Self; 32] {
        simd.to_array()
    }

    fn simd64_to_arr(simd: Self::Simd64) -> [Self; 64] {
        simd.to_array()
    }
}