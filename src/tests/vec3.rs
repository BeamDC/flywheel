use crate::math::vec3::Vec3;

#[test]
fn test_dot() {
    assert_eq!(Vec3::X.dot(Vec3::Y), 0.0)
}

#[test]
fn test_cross() {
    assert_eq!(Vec3::X.cross(Vec3::Y), Vec3::Z)
}

#[test]
fn test_mag() {
    assert_eq!(Vec3(1.0, 0.0, 0.0).map(|v| v * 3.).magnitude(), 3.0)
}

#[test]
fn test_normalize() {
    assert_eq!(Vec3::X.normalize(), Vec3::X);
}