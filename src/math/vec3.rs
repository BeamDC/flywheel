use std::fmt::{Debug, Formatter};

/// A three-dimensional vector
#[derive(Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// wrapper function for [`Vec3::new()`]
#[inline(always)]
#[allow(non_snake_case)]
pub const fn Vec3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

impl Vec3 {
    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }

    /// a vector where both x and y components are zero
    pub const ZERO: Vec3 = Vec3(0.0, 0.0, 0.0);

    /// a unit vector parallel to the x-axis, in the positive direction
    pub const X: Vec3 = Vec3(1.0, 0.0, 0.0);

    /// a unit vector parallel to the y-axis, in the positive direction
    pub const Y: Vec3 = Vec3(0.0, 1.0, 0.0);

    /// a unit vector parallel to the z-axis, in the positive direction
    pub const Z: Vec3 = Vec3(0.0, 0.0, 1.0);

    /// a unit vector parallel to the x-axis, in the negative direction
    pub const NEG_X: Vec3 = Vec3(-1.0, 0.0, 0.0);

    /// a unit vector parallel to the y-axis, in the negative direction
    pub const NEG_Y: Vec3 = Vec3(0.0, -1.0, 0.0);

    /// a unit vector parallel to the y-axis, in the negative direction
    pub const NEG_Z: Vec3 = Vec3(0.0, 0.0, -1.0);

    /// check whether a vectors components are finite values
    #[inline(always)]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// computes the dot product of two [`vec3`]s
    #[inline(always)]
    pub fn dot(self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline(always)]
    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// computes the magnitude of a [`vec3`]
    #[inline(always)]
    pub fn magnitude(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// normalize a [`vec3`]
    #[inline(always)]
    pub fn normalize(self) -> Vec3 {
        let mag = self.magnitude();
        let Vec3 { x, y, z } = self;
        let x = if x == 0.0 { 0.0 } else { mag / x };
        let y = if y == 0.0 { 0.0 } else { mag / y };
        let z = if z == 0.0 { 0.0 } else { mag / z };
        let res = Vec3(x ,y, z);
        assert!(res.is_finite());
        res
    }

    /// apple some function `f` for the x, y, and z components of a [`Vec3`]
    #[inline]
    pub fn map<F>(&mut self, f: F) -> Vec3
    where
        F: Fn(f32) -> f32
    {
        Vec3(f(self.x), f(self.y), f(self.z))
    }
}

impl Debug for Vec3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "<{}, {}, {}>", self.x, self.y, self.z)
    }
}