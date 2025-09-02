use crate::math::vec2::Vec2;

#[test]
fn test_dot() {
    assert_eq!(Vec2::X.dot(Vec2::Y), 0.0)
}

#[test]
fn test_mag() {
    assert_eq!(Vec2(1.0, 0.0).map(|v| v * 3.).magnitude(), 3.0)
}

#[test]
fn test_normalize() {
    assert_eq!(Vec2::X.normalize(), Vec2::X);
}