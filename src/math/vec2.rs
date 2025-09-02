use std::fmt::{Debug, Formatter};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

/// A two-dimensional vector
#[derive(Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

/// wrapper function for [`Vec2::new()`]
#[inline(always)]
#[allow(non_snake_case)]
pub const fn Vec2(x: f32, y: f32) -> Vec2 {
    Vec2::new(x, y)
}

impl Vec2 {
    /// construct a new [`Vec2`]
    #[inline(always)]
    pub const fn new(x: f32, y: f32) -> Vec2 {
        Vec2 { x, y }
    }

    /// a vector where both x and y components are zero
    pub const ZERO: Vec2 = Vec2(0.0, 0.0);

    /// a unit vector parallel to the x-axis, in the positive direction
    pub const X: Vec2 = Vec2(1.0, 0.0);

    /// a unit vector parallel to the y-axis, in the positive direction
    pub const Y: Vec2 = Vec2(0.0, 1.0);

    /// a unit vector parallel to the x-axis, in the negative direction
    pub const NEG_X: Vec2 = Vec2(-1.0, 0.0);

    /// a unit vector parallel to the y-axis, in the negative direction
    pub const NEG_Y: Vec2 = Vec2(0.0, -1.0);

    /// check whether a vectors components are finite values
    #[inline(always)]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    /// computes the dot product of two [`vec2`]s
    #[inline(always)]
    pub fn dot(self, other: Vec2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// computes the magnitude of a [`vec2`]
    #[inline(always)]
    pub fn magnitude(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// normalize a [`vec2`]
    #[inline(always)]
    pub fn normalize(self) -> Vec2 {
        let mag = self.magnitude();

        let res = match self {
            Vec2::ZERO => Vec2::ZERO,
            Vec2 { x, y: 0.0 } => Vec2(mag / x, 0.0),
            Vec2 { x: 0.0, y } => Vec2(0.0, mag / y),
            _ => Vec2(mag / self.x, mag / self.y)
        };
        assert!(res.is_finite());
        res
    }

    /// apple some function `f` for both the x and y components of a [`Vec2`]
    #[inline]
    pub fn map<F>(&mut self, f: F) -> Vec2
    where
        F: Fn(f32) -> f32
    {
        Vec2(f(self.x), f(self.y))
    }
}

impl Debug for Vec2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "<{}, {}>", self.x, self.y)
    }
}

impl Add for Vec2 {
    type Output = Vec2;

    fn add(self, rhs: Self) -> Self::Output {
        Vec2(
            self.x + rhs.x,
            self.y + rhs.y,
        )
    }
}

impl AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl Sub for Vec2 {
    type Output = Vec2;

    fn sub(self, rhs: Self) -> Self::Output {
        Vec2(
            self.x - rhs.x,
            self.y - rhs.y,
        )
    }
}

impl SubAssign for Vec2 {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl Mul<f32> for Vec2 {
    type Output = Vec2;

    fn mul(self, rhs: f32) -> Self::Output {
        Vec2(
            self.x * rhs,
            self.y * rhs,
        )
    }
}

impl MulAssign for Vec2 {
    fn mul_assign(&mut self, rhs: Self) {
        self.x *= rhs.x;
        self.y *= rhs.y;
    }
}

impl Div for Vec2 {
    type Output = Vec2;

    fn div(self, rhs: Self) -> Self::Output {
        Vec2(
            self.x / rhs.x,
            self.y / rhs.y,
        )
    }
}

impl DivAssign for Vec2 {
    fn div_assign(&mut self, rhs: Self) {
        self.x /= rhs.x;
        self.y /= rhs.y;
    }
}