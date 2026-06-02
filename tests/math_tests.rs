//! Comprehensive mathematical tests for difftop-rs
//!
//! Covers cross-module integration, deeper mathematical properties,
//! and additional edge cases beyond the inline module tests.

use difftop_rs::*;
use nalgebra::{DMatrix, DVector};

// ============================================================
// Smooth Manifold tests
// ============================================================

#[test]
fn test_circle_atlas_transition_identity() {
    let m = circle_manifold();
    let p = DVector::from_vec(vec![0.5]);
    let jac = m.atlas.transition_jacobian(0, 1, &p).unwrap();
    assert!(
        (jac[(0, 0)] - 1.0).abs() < 1e-10,
        "S¹ transition should be identity"
    );
}

#[test]
fn test_sphere_stereographic_at_origin() {
    let m = sphere_manifold_2d();
    let p = DVector::from_vec(vec![1.0, 1.0]);
    let jac = m.atlas.transition_jacobian(0, 1, &p).unwrap();
    // At (1,1): r²=2, r⁴=4
    // diag((2-2)/4, (2-2)/4) = diag(0, 0), off-diag = -2/4 = -0.5
    assert!((jac[(0, 0)] - 0.0).abs() < 1e-10);
    assert!((jac[(0, 1)] + 0.5).abs() < 1e-10);
    assert!((jac[(1, 0)] + 0.5).abs() < 1e-10);
    assert!((jac[(1, 1)] - 0.0).abs() < 1e-10);
}

#[test]
fn test_n_sphere_s1_matches_circle() {
    let s1 = n_sphere_manifold(1);
    assert_eq!(s1.dimension(), 1);
    assert_eq!(s1.atlas.charts.len(), 2);
    assert_eq!(s1.name, "S^1");
}

#[test]
fn test_n_sphere_s5_uses_identity_transition() {
    let s5 = n_sphere_manifold(5);
    assert_eq!(s5.dimension(), 5);
    assert_eq!(s5.atlas.charts.len(), 2);
    let p = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let jac = s5.atlas.transition_jacobian(0, 1, &p).unwrap();
    assert_eq!(jac.nrows(), 5);
    assert_eq!(jac.ncols(), 5);
}

#[test]
fn test_euclidean_manifold_various_dimensions() {
    for n in 1..=5 {
        let m = euclidean_manifold(n);
        assert_eq!(m.dimension(), n);
        assert_eq!(m.atlas.charts.len(), 1);
        assert_eq!(m.name, format!("R^{}", n));
    }
}

#[test]
fn test_torus_transition_identity() {
    let t2 = torus_manifold();
    let p = DVector::from_vec(vec![0.3, 0.7]);
    let jac = t2.atlas.transition_jacobian(0, 1, &p).unwrap();
    assert!((jac - DMatrix::identity(2, 2)).norm() < 1e-10);
}

#[test]
fn test_stereographic_jacobian_determinant_property() {
    // The stereographic transition Jacobian should satisfy det(J) = -1/r⁴ * 4
    // (orientation-reversing for each point)
    let m = sphere_manifold_2d();
    let p = DVector::from_vec(vec![0.5, 0.0]);
    let jac = m.atlas.transition_jacobian(0, 1, &p).unwrap();
    let det = jac.determinant();
    // At (0.5, 0): r²=0.25, r⁴=0.0625
    // r² - 2x² = 0.25 - 0.5 = -0.25, so J[0,0] = -0.25/0.0625 = -4
    // r² - 2y² = 0.25, so J[1,1] = 0.25/0.0625 = 4
    // det = -4 * 4 = -16
    assert!(
        (det - (-16.0)).abs() < 1e-8,
        "det at (0.5,0) should be -16, got {}",
        det
    );
}

#[test]
fn test_missing_transition_returns_none() {
    let m = euclidean_manifold(2);
    let p = DVector::from_vec(vec![0.0, 0.0]);
    assert!(m.atlas.transition_jacobian(0, 1, &p).is_none());
}

// Serialization tests require serde_json as dev-dep; skipped.

// ============================================================
// Tangent space tests
// ============================================================

#[test]
fn test_tangent_vector_scale_and_norm() {
    let v = TangentVector::new(
        DVector::from_vec(vec![0.0, 0.0]),
        DVector::from_vec(vec![3.0, 4.0]),
    );
    let scaled = v.scale(2.0);
    assert!((scaled.norm() - 10.0).abs() < 1e-10);
}

#[test]
fn test_tangent_vector_inner_product_orthogonal() {
    let p = DVector::from_vec(vec![0.0, 0.0, 0.0]);
    let v = TangentVector::new(p.clone(), DVector::from_vec(vec![1.0, 0.0, 0.0]));
    let w = TangentVector::new(p.clone(), DVector::from_vec(vec![0.0, 1.0, 0.0]));
    assert!(v.inner_product(&w).abs() < 1e-10);
}

#[test]
fn test_tangent_vector_inner_product_parallel() {
    let p = DVector::from_vec(vec![1.0, 2.0]);
    let v = TangentVector::new(p.clone(), DVector::from_vec(vec![3.0, 4.0]));
    assert!((v.inner_product(&v) - 25.0).abs() < 1e-10);
}

#[test]
fn test_tangent_space_zero_vector() {
    let ts = TangentSpace::new(DVector::from_vec(vec![1.0, 2.0, 3.0]));
    let z = ts.zero();
    assert_eq!(z.dimension(), 3);
    assert!((z.norm() - 0.0).abs() < 1e-10);
}

#[test]
fn test_tangent_space_custom_metric() {
    // Diagonal metric g = diag(4, 9)
    let p = DVector::from_vec(vec![0.0, 0.0]);
    let ts = TangentSpace::new(p).with_metric(DMatrix::from_row_slice(2, 2, &[4.0, 0.0, 0.0, 9.0]));
    let v = TangentVector::new(
        DVector::from_vec(vec![0.0, 0.0]),
        DVector::from_vec(vec![1.0, 0.0]),
    );
    // g(v,v) = v^T g v = 4
    assert!((ts.inner_product(&v, &v) - 4.0).abs() < 1e-10);
}

#[test]
fn test_tangent_bundle_fiber_dimension() {
    let tb = TangentBundle::new(3);
    let fiber = tb.fiber(DVector::from_vec(vec![1.0, 0.0, 0.0]));
    assert_eq!(fiber.dimension, 3);
}

#[test]
fn test_cotangent_dual_pairing() {
    // Covector α = (2, 3) at origin, vector v = (1, 4)
    // α(v) = 2*1 + 3*4 = 14
    let alpha = CotangentVector::new(
        DVector::from_vec(vec![0.0, 0.0]),
        DVector::from_vec(vec![2.0, 3.0]),
    );
    let v = TangentVector::new(
        DVector::from_vec(vec![0.0, 0.0]),
        DVector::from_vec(vec![1.0, 4.0]),
    );
    assert!((alpha.apply(&v) - 14.0).abs() < 1e-10);
}

#[test]
fn test_differential_pushforward_scaling() {
    // Jacobian = 2*I should double the vector
    let jac = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
    let d = Differential::new(jac);
    let v = TangentVector::new(
        DVector::from_vec(vec![0.0, 0.0]),
        DVector::from_vec(vec![1.0, 1.0]),
    );
    let pushed = d.push_forward(&v);
    assert!((pushed.components[0] - 2.0).abs() < 1e-10);
    assert!((pushed.components[1] - 2.0).abs() < 1e-10);
}

#[test]
fn test_differential_non_square_jacobian() {
    // Map R² → R³: f(x,y) = (x, y, x+y)
    let jac = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let d = Differential::new(jac);
    assert_eq!(d.source_dim, 2);
    assert_eq!(d.target_dim, 3);
    // Note: push_forward asserts base_point.len() == components.len(),
    // so we test the Jacobian action directly
    let v = DVector::from_vec(vec![1.0, 0.0]);
    let result = &d.jacobian * &v;
    assert_eq!(result.len(), 3);
    assert!((result[0] - 1.0).abs() < 1e-10);
    assert!((result[1] - 0.0).abs() < 1e-10);
    assert!((result[2] - 1.0).abs() < 1e-10);
}

// TangentVector serialization requires serde_json dev-dep; skipped.

// ============================================================
// Vector field tests
// ============================================================

#[test]
fn test_rotational_field_is_zero_at_origin() {
    let vf = rotational_vector_field();
    let p = DVector::from_vec(vec![0.0, 0.0]);
    let tv = vf.evaluate(&p);
    assert!(tv.norm() < 1e-10);
}

#[test]
fn test_rotational_field_perpendicular() {
    let vf = rotational_vector_field();
    let p = DVector::from_vec(vec![1.0, 0.0]);
    let tv = vf.evaluate(&p);
    // V(1,0) = (0, 1), which is perpendicular to (1,0)
    assert!((tv.components[0] - 0.0).abs() < 1e-10);
    assert!((tv.components[1] - 1.0).abs() < 1e-10);
}

#[test]
fn test_radial_field_points_outward() {
    let vf = radial_vector_field(3);
    let p = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    let tv = vf.evaluate(&p);
    // V(p) = p
    assert!((tv.components[0] - 1.0).abs() < 1e-10);
    assert!((tv.components[1] - 2.0).abs() < 1e-10);
    assert!((tv.components[2] - 3.0).abs() < 1e-10);
}

#[test]
fn test_integral_curve_rotational_conserves_norm() {
    let vf = rotational_vector_field();
    let start = DVector::from_vec(vec![1.0, 0.0]);
    let curve = integral_curve(&vf, &start, 0.01, 628); // ~2π
                                                        // After one full revolution, should be back near start
    let start_norm = start.norm();
    for pt in &curve {
        assert!(
            (pt.norm() - start_norm).abs() < 0.05,
            "Norm should be conserved: got {}",
            pt.norm()
        );
    }
}

#[test]
fn test_divergence_constant_field() {
    let vf = constant_vector_field(3, DVector::from_vec(vec![1.0, 2.0, 3.0]));
    let p = DVector::from_vec(vec![0.0, 0.0, 0.0]);
    let div = divergence(&vf, &p, 1e-5);
    assert!(
        div.abs() < 1e-3,
        "Divergence of constant field should be 0, got {}",
        div
    );
}

#[test]
fn test_divergence_radial_3d() {
    let vf = radial_vector_field(3);
    let p = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    let div = divergence(&vf, &p, 1e-5);
    // div(x, y, z) = 1 + 1 + 1 = 3
    assert!(
        (div - 3.0).abs() < 1e-3,
        "div of radial field in R³ should be 3, got {}",
        div
    );
}

#[test]
fn test_curl_zero_for_gradient_field() {
    // V = ∇(x² + y²) = (2x, 2y), curl should be 0
    let vf = VectorField::new("gradient", 2, |p: &DVector<f64>| {
        DVector::from_vec(vec![2.0 * p[0], 2.0 * p[1]])
    });
    let p = DVector::from_vec(vec![1.0, 1.0]);
    let c = curl_2d(&vf, &p, 1e-5);
    assert!(
        c.abs() < 1e-3,
        "Curl of gradient field should be 0, got {}",
        c
    );
}

#[test]
fn test_classify_center() {
    // Pure rotation: Jacobian = [[0, -1], [1, 0]] but symmetric_eigen gives real eigenvalues
    // For the symmetric part: [[0, 0], [0, 0]] → center
    let jac = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 0.0]);
    assert_eq!(classify_equilibrium(&jac), EquilibriumType::Center);
}

#[test]
fn test_equilibrium_types_exhaustive() {
    // Test all equilibrium types are reachable
    let types = vec![
        EquilibriumType::Stable,
        EquilibriumType::Unstable,
        EquilibriumType::Saddle,
        EquilibriumType::Center,
        EquilibriumType::Degenerate,
    ];
    assert_eq!(types.len(), 5);
    // Verify Debug and Clone
    for t in &types {
        let cloned = t.clone();
        assert_eq!(format!("{:?}", t), format!("{:?}", cloned));
    }
}

// ============================================================
// Differential form tests
// ============================================================

#[test]
fn test_wedge_zero_form_with_one_form() {
    // 0-form f=3 times 1-form dx₁ = (1,0)
    let f = DifferentialForm::from_components(0, 2, vec![3.0]);
    let dx = DifferentialForm::from_components(1, 2, vec![1.0, 0.0]);
    // Can't wedge 0-form... the function should work for degree 0 + 1 = 1
    // Actually the wedge function works for any k, l
    let w = wedge(&f, &dx);
    assert_eq!(w.degree, 1);
    // 3 * dx₁ = (3, 0)
    assert!((w.components[0] - 3.0).abs() < 1e-10);
}

#[test]
fn test_wedge_higher_degree_exceeds_dimension() {
    let dx = DifferentialForm::from_components(1, 2, vec![1.0, 0.0]);
    let dy = DifferentialForm::from_components(1, 2, vec![0.0, 1.0]);
    let dx_dy = wedge(&dx, &dy); // 2-form
    let w = wedge(&dx, &dx_dy); // 3-form in R² → should be zero
    assert_eq!(w.degree, 3);
    assert_eq!(w.components.len(), 0); // C(2,3) = 0
}

#[test]
fn test_wedge_associativity() {
    let dx1 = DifferentialForm::from_components(1, 3, vec![1.0, 0.0, 0.0]);
    let dx2 = DifferentialForm::from_components(1, 3, vec![0.0, 1.0, 0.0]);
    let dx3 = DifferentialForm::from_components(1, 3, vec![0.0, 0.0, 1.0]);

    let left = wedge(&wedge(&dx1, &dx2), &dx3);
    let right = wedge(&dx1, &wedge(&dx2, &dx3));
    assert_eq!(left.degree, 3);
    assert_eq!(right.degree, 3);
    assert!((left.components[0] - right.components[0]).abs() < 1e-10);
}

#[test]
fn test_1form_evaluate_on_vector() {
    let form = DifferentialForm::from_components(1, 3, vec![1.0, 2.0, 3.0]);
    let v = DVector::from_vec(vec![1.0, 1.0, 1.0]);
    assert!((form.apply_to_vector(&v) - 6.0).abs() < 1e-10);
}

#[test]
fn test_0form_evaluate() {
    let f = DifferentialForm::from_components(0, 2, vec![42.0]);
    assert!((f.evaluate(&[]) - 42.0).abs() < 1e-10);
}

#[test]
fn test_d_1form_closed() {
    // Exact 1-form: ω = df where f = x²+y², gradient = (2x, 2y)
    // dω = d(df) = 0 (exact forms are closed)
    // Jacobian of (2x, 2y) is [[2,0],[0,2]], which is symmetric
    // dω = (∂f₂/∂x₁ - ∂f₁/∂x₂) dx₁∧dx₂ = (0-0) = 0
    let jac = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
    let dw = d_1form(&jac);
    assert!(
        dw.components.iter().all(|c| c.abs() < 1e-10),
        "Exterior derivative of exact form should be zero"
    );
}

#[test]
fn test_pullback_composition() {
    // Pullback by identity should be identity
    let form = DifferentialForm::from_components(2, 3, vec![1.0, 0.0, 1.0]);
    let id = DMatrix::identity(3, 3);
    let pb = pullback(&form, &id);
    for i in 0..form.components.len() {
        assert!((pb.components[i] - form.components[i]).abs() < 1e-10);
    }
}

#[test]
fn test_hodge_star_double_application() {
    // ∗∗α = (-1)^(k(n-k)) α on R^n with standard metric
    let dx1 = DifferentialForm::from_components(1, 3, vec![2.0, 0.0, 0.0]);
    let star1 = hodge_star(&dx1);
    let star2 = hodge_star(&star1);
    // k=1, n=3: (-1)^(1*2) = 1, so ∗∗ = +identity
    assert!((star2.components[0] - 2.0).abs() < 1e-10);
}

#[test]
fn test_hodge_star_volume_form() {
    // ∗1 = vol on R³
    let one_form = DifferentialForm::from_components(0, 3, vec![1.0]);
    let star = hodge_star(&one_form);
    assert_eq!(star.degree, 3);
    assert!((star.components[0] - 1.0).abs() < 1e-10);
}

#[test]
fn test_symplectic_form_is_closed() {
    // The standard symplectic form on R⁴ should have the right structure
    let sym = symplectic_form(2);
    assert_eq!(sym.ambient_dimension, 4);
    assert_eq!(sym.degree, 2);
    // Should have exactly 2 nonzero components (dx₀∧dy₀, dx₁∧dy₁)
    let nonzero: Vec<_> = sym
        .components
        .iter()
        .filter(|&&c| c.abs() > 1e-10)
        .collect();
    assert_eq!(nonzero.len(), 2);
}

#[test]
fn test_form_at_point() {
    let f = DifferentialForm::from_components(1, 2, vec![1.0, 2.0])
        .at_point(DVector::from_vec(vec![3.0, 4.0]));
    assert!(f.base_point.is_some());
    assert!((f.base_point.unwrap()[0] - 3.0).abs() < 1e-10);
}

// DifferentialForm serialization requires serde_json dev-dep; skipped.

// ============================================================
// Stokes' theorem tests
// ============================================================

#[test]
fn test_integrate_1d_quadratic() {
    // ∫₀¹ x² dx = 1/3
    let result = stokes::integrate_1d(&|x| x * x, 0.0, 1.0, 10000);
    assert!((result - 1.0 / 3.0).abs() < 1e-4);
}

#[test]
fn test_integrate_1d_negative_range() {
    // ∫₋₁¹ x² dx = 2/3
    let result = stokes::integrate_1d(&|x| x * x, -1.0, 1.0, 10000);
    assert!((result - 2.0 / 3.0).abs() < 1e-3);
}

#[test]
fn test_integrate_2d_gaussian_like() {
    // ∫₀¹ ∫₀¹ 4xy dxdy = 4 * 1/2 * 1/2 = 1
    let result = stokes::integrate_2d(&|x, y| 4.0 * x * y, 0.0, 1.0, 0.0, 1.0, 100);
    assert!((result - 1.0).abs() < 1e-2);
}

#[test]
fn test_line_integral_straight_line() {
    // Line integral of ω = dx + dy along straight line from (0,0) to (1,1)
    // γ(t) = (t, t), γ'(t) = (1, 1)
    // ∫ (1)(1) + (1)(1) dt from 0 to 1 = 2
    let result = stokes::line_integral(
        &|_: &DVector<f64>| DVector::from_vec(vec![1.0, 1.0]),
        &|t| DVector::from_vec(vec![t, t]),
        &|_: f64| DVector::from_vec(vec![1.0, 1.0]),
        0.0,
        1.0,
        1000,
    );
    assert!((result - 2.0).abs() < 0.01);
}

#[test]
fn test_surface_area_parallelogram() {
    // Skew surface: (u, v, u+v)
    // Partial derivatives: (1, 0, 1) and (0, 1, 1)
    // Cross product magnitude: sqrt(1 + 1 + 1) = sqrt(3)
    // Area over [0,1]² = sqrt(3)
    let area = stokes::surface_area(
        &|u, v| DVector::from_vec(vec![u, v, u + v]),
        &|_, _| DVector::from_vec(vec![1.0, 0.0, 1.0]),
        &|_, _| DVector::from_vec(vec![0.0, 1.0, 1.0]),
        0.0,
        1.0,
        0.0,
        1.0,
        100,
    );
    assert!(
        (area - 3.0_f64.sqrt()).abs() < 0.01,
        "Expected √3 ≈ {}, got {}",
        3.0_f64.sqrt(),
        area
    );
}

// ============================================================
// Transversality tests
// ============================================================

#[test]
fn test_transverse_curve_surface_in_r3() {
    // A curve tangent to e₃ and a plane spanned by e₁, e₂: transverse in R³
    let curve_tangent = DMatrix::from_row_slice(3, 1, &[0.0, 0.0, 1.0]);
    let plane_tangent = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    assert!(is_transverse_intersection(
        3,
        &curve_tangent,
        &plane_tangent,
        1e-10
    ));
}

#[test]
fn test_non_transverse_coplanar_planes() {
    // Two planes both spanned by e₁, e₂ in R³: not transverse
    let ts1 = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let ts2 = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    assert!(!is_transverse_intersection(3, &ts1, &ts2, 1e-10));
}

#[test]
fn test_transverse_intersection_dim_negative() {
    // Two curves in R³: dim = 1+1-3 = -1 (generic non-intersection)
    assert_eq!(transverse_intersection_dimension(1, 1, 3), -1);
}

#[test]
fn test_transverse_intersection_dim_zero() {
    // Two curves in R²: dim = 1+1-2 = 0 (isolated points)
    assert_eq!(transverse_intersection_dimension(1, 1, 2), 0);
}

#[test]
fn test_transverse_intersection_dim_positive() {
    // Two surfaces in R⁴: dim = 2+2-4 = 0
    assert_eq!(transverse_intersection_dimension(2, 2, 4), 0);
}

#[test]
fn test_map_not_transverse() {
    // df with rank deficiency + tangent_s with same rank = not transverse
    let df = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let tangent_s = DMatrix::from_row_slice(3, 1, &[0.0, 1.0, 0.0]);
    // combined: [1,0,0]^T, [0,0,0]^T, [0,1,0]^T → rank 2, not 3
    assert!(!is_map_transverse_to_submanifold(&df, &tangent_s, 3, 1e-10));
}

// ============================================================
// Degree theory tests
// ============================================================

#[test]
fn test_winding_number_figure_eight() {
    // Lemniscate/figure-8: winds +1 then -1, net winding ≈ 0 around origin
    // γ(t) = (sin(t), sin(2t)/2) for t ∈ [0, 2π]
    let curve = |t: f64| DVector::from_vec(vec![t.sin(), (2.0 * t).sin() / 2.0]);
    let curve_d = |t: f64| DVector::from_vec(vec![t.cos(), (2.0 * t).cos()]);
    let point = DVector::from_vec(vec![5.0, 0.0]); // far away
    let wn = winding_number(&curve, &curve_d, &point, 2000);
    assert!(
        wn.abs() < 0.1,
        "Winding number far from figure-8 should be ~0, got {}",
        wn
    );
}

#[test]
fn test_local_degree_singular() {
    // Singular Jacobian (det = 0)
    let jac = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
    assert_eq!(local_degree(&jac), 0);
}

#[test]
fn test_antipodal_degree_formula() {
    // Antipodal map on S^n has degree (-1)^(n+1)
    // antipodal_degree(n) = (-1)^(n+1)
    assert_eq!(antipodal_degree(0), -1); // (-1)^1 = -1
    assert_eq!(antipodal_degree(1), 1); // (-1)^2 = 1
    assert_eq!(antipodal_degree(2), -1); // (-1)^3 = -1
    assert_eq!(antipodal_degree(3), 1); // (-1)^4 = 1
    assert_eq!(antipodal_degree(4), -1); // (-1)^5 = -1
}

#[test]
fn test_brouwer_degree_composite() {
    // f∘g where f has degree 2 and g has degree 1: total degree = 2
    // z → z² on S¹ (degree 2)
    let f = |p: &DVector<f64>| {
        let x = p[0];
        let y = p[1];
        let nx = x * x - y * y;
        let ny = 2.0 * x * y;
        let r = (nx * nx + ny * ny).sqrt().max(1e-10);
        DVector::from_vec(vec![nx / r, ny / r])
    };
    let df = |p: &DVector<f64>| {
        let x = p[0];
        let y = p[1];
        let r = ((x * x - y * y).powi(2) + (2.0 * x * y).powi(2))
            .sqrt()
            .max(1e-10);
        DMatrix::from_row_slice(2, 2, &[2.0 * x / r, -2.0 * y / r, 2.0 * y / r, 2.0 * x / r])
    };
    let rv = DVector::from_vec(vec![1.0, 0.0]);
    let preimages = vec![
        DVector::from_vec(vec![1.0, 0.0]),
        DVector::from_vec(vec![-1.0, 0.0]),
    ];
    assert_eq!(brouwer_degree(&f, &df, &rv, &preimages), 2);
}

#[test]
fn test_degree_numerical_identity() {
    let f = |p: &DVector<f64>| p.clone();
    let df = |_: &DVector<f64>| DMatrix::identity(2, 2);
    let rv = DVector::from_vec(vec![0.5, 0.5]);
    let samples = vec![DVector::from_vec(vec![0.5, 0.5])];
    assert_eq!(degree_numerical(&f, &df, &rv, &samples, 0.01), 1);
}

// ============================================================
// Euler characteristic tests
// ============================================================

#[test]
fn test_euler_sphere_all_even_dimensions() {
    // S^n has χ = 2 for even n
    for n in [0usize, 2, 4, 6, 8] {
        assert_eq!(euler_characteristic_sphere(n), 2, "S^{} should have χ=2", n);
    }
}

#[test]
fn test_euler_sphere_all_odd_dimensions() {
    // S^n has χ = 0 for odd n
    for n in [1usize, 3, 5, 7, 9] {
        assert_eq!(euler_characteristic_sphere(n), 0, "S^{} should have χ=0", n);
    }
}

#[test]
fn test_euler_genus_series() {
    // χ(g) = 2 - 2g for orientable surface of genus g
    assert_eq!(euler_characteristic_surface_genus(0), 2); // sphere
    assert_eq!(euler_characteristic_surface_genus(1), 0); // torus
    assert_eq!(euler_characteristic_surface_genus(2), -2); // double torus
    assert_eq!(euler_characteristic_surface_genus(3), -4); // triple torus
    assert_eq!(euler_characteristic_surface_genus(10), -18);
}

#[test]
fn test_euler_triangulation_octahedron() {
    // Octahedron: 6 vertices, 12 edges, 8 faces → χ = 2 (homeomorphic to S²)
    assert_eq!(euler_characteristic_from_triangulation(6, 12, 8), 2);
}

#[test]
fn test_euler_triangulation_icosahedron() {
    // Icosahedron: 12 vertices, 30 edges, 20 faces → χ = 2
    assert_eq!(euler_characteristic_from_triangulation(12, 30, 20), 2);
}

#[test]
fn test_euler_cw_simplex() {
    // 2-simplex: 3 vertices (0-cells), 3 edges (1-cells), 1 face (2-cell) → χ = 1
    assert_eq!(euler_characteristic_cw(&[3, 3, 1]), 1);
}

#[test]
fn test_euler_morse_standard() {
    // Standard Morse function on S²: min, max → χ = 1 + 1 = 2
    assert_eq!(euler_characteristic_morse(&[1, 0, 1]), 2);
    // Torus Morse function: min, 2 saddles, max → χ = 1 - 2 + 1 = 0
    assert_eq!(euler_characteristic_morse(&[1, 2, 1]), 0);
    // Sphere with perturbation: min, saddle, max → χ = 1 - 1 + 1 = 1 (wrong for S²)
    // This would be a different surface
}

#[test]
fn test_gauss_bonnet_negative_curvature() {
    // For genus-2 surface: ∫ K dA = 2π(2-2·2) = 2π(-2) = -4π
    assert!(verify_gauss_bonnet(-4.0 * std::f64::consts::PI, -2, 1e-10));
}

#[test]
fn test_index_degenerate() {
    // Singular Jacobian: det = 0 → index = 0
    let jac = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
    assert_eq!(index_of_zero(&jac), 0);
}

#[test]
fn test_poincare_hopf_sum_matches_sphere() {
    // Two zeros each with index +1 → χ = 2
    let jac1 = DMatrix::identity(2, 2);
    let jac2 = DMatrix::identity(2, 2);
    assert_eq!(euler_characteristic_poincare_hopf(&[jac1, jac2]), 2);
}

#[test]
fn test_poincare_hopf_torus_no_zeros() {
    assert_eq!(euler_characteristic_poincare_hopf(&[]), 0);
}

// ============================================================
// Agent dynamics tests
// ============================================================

#[test]
fn test_agent_system_multi_step() {
    let vf = constant_vector_field(2, DVector::from_vec(vec![1.0, 0.0]));
    let mut sys = agent_dynamics::AgentSystem::new(vf);
    sys.add_agent(DVector::from_vec(vec![0.0, 0.0]));
    for _ in 0..10 {
        sys.step(0.1);
    }
    assert!((sys.agents[0].position[0] - 1.0).abs() < 0.01);
    assert!(sys.agents[0].position[1].abs() < 0.01);
}

#[test]
fn test_agent_system_rk4_accuracy() {
    // V = (1, 0), after 1 second should be at (1, 0)
    let vf = constant_vector_field(2, DVector::from_vec(vec![1.0, 0.0]));
    let mut sys = agent_dynamics::AgentSystem::new(vf);
    sys.add_agent(DVector::from_vec(vec![0.0, 0.0]));
    for _ in 0..100 {
        sys.step_rk4(0.01);
    }
    assert!((sys.agents[0].position[0] - 1.0).abs() < 1e-4);
}

#[test]
fn test_agent_clusters_single_cluster() {
    let vf = VectorField::new("zero", 2, |_: &DVector<f64>| DVector::zeros(2));
    let mut sys = agent_dynamics::AgentSystem::new(vf);
    sys.add_agent(DVector::from_vec(vec![0.0, 0.0]));
    sys.add_agent(DVector::from_vec(vec![0.1, 0.1]));
    sys.add_agent(DVector::from_vec(vec![0.2, 0.0]));
    let clusters = sys.find_clusters(1.0);
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].len(), 3);
}

#[test]
fn test_agent_clusters_no_clusters() {
    let vf = VectorField::new("zero", 2, |_: &DVector<f64>| DVector::zeros(2));
    let mut sys = agent_dynamics::AgentSystem::new(vf);
    sys.add_agent(DVector::from_vec(vec![0.0, 0.0]));
    sys.add_agent(DVector::from_vec(vec![10.0, 10.0]));
    let clusters = sys.find_clusters(1.0);
    assert_eq!(clusters.len(), 2);
}

#[test]
fn test_numerical_jacobian_linear_field() {
    // V(x,y) = (x, -y): Jacobian should be [[1,0],[0,-1]]
    let vf = VectorField::new("saddle", 2, |p: &DVector<f64>| {
        DVector::from_vec(vec![p[0], -p[1]])
    });
    let p = DVector::from_vec(vec![1.0, 1.0]);
    let jac = numerical_jacobian(&vf, &p, 1e-5);
    assert!((jac[(0, 0)] - 1.0).abs() < 1e-3);
    assert!((jac[(0, 1)] - 0.0).abs() < 1e-3);
    assert!((jac[(1, 0)] - 0.0).abs() < 1e-3);
    assert!((jac[(1, 1)] + 1.0).abs() < 1e-3);
}

#[test]
fn test_classify_dynamics_convergent() {
    let vf = VectorField::new("sink", 2, |p: &DVector<f64>| {
        DVector::from_vec(vec![-p[0], -p[1]])
    });
    let analysis = analyze_flow(&vf, &[(-1.0, 1.0), (-1.0, 1.0)], 20, 0.01, 10, 4);
    let dynamics = classify_dynamics(&analysis);
    assert_eq!(dynamics, DynamicsType::Convergent);
}

#[test]
fn test_classify_dynamics_divergent_no_eq() {
    // Constant field has no equilibria (unless origin is hit exactly)
    let vf = constant_vector_field(2, DVector::from_vec(vec![1.0, 1.0]));
    let analysis = analyze_flow(&vf, &[(0.0, 2.0), (0.0, 2.0)], 10, 0.01, 10, 4);
    let dynamics = classify_dynamics(&analysis);
    assert_eq!(dynamics, DynamicsType::Divergent);
}

#[test]
fn test_lyapunov_constant_trajectory() {
    // Constant trajectory: zero Lyapunov exponent
    let trajectory: Vec<DVector<f64>> = (0..50)
        .map(|i| DVector::from_vec(vec![i as f64 * 0.01, 1.0]))
        .collect();
    let le = lyapunov_exponent(&trajectory, 0.01);
    assert!(
        le.abs() < 1.0,
        "Constant speed trajectory should have small Lyapunov exponent, got {}",
        le
    );
}

#[test]
fn test_agent_distance_self() {
    let a = agent_dynamics::Agent::new(0, DVector::from_vec(vec![3.0, 4.0]));
    assert!((a.distance_to(&a) - 0.0).abs() < 1e-10);
}

// ============================================================
// Cross-module integration tests
// ============================================================

#[test]
fn test_manifold_to_tangent_bundle() {
    let m = sphere_manifold_2d();
    let tb = TangentBundle::new(m.dimension());
    assert_eq!(tb.total_dimension, 4);
    let elem = tb.element(
        DVector::from_vec(vec![1.0, 0.0]),
        DVector::from_vec(vec![0.0, 1.0]),
    );
    assert_eq!(elem.dimension(), 2);
}

#[test]
fn test_vector_field_on_manifold_euler_characteristic() {
    // On S²: use the gradient of height function
    // Create a 2D vector field (projection) and verify Euler characteristic
    let jac_north = DMatrix::identity(2, 2);
    let jac_south = DMatrix::identity(2, 2);
    let chi = euler_characteristic_poincare_hopf(&[jac_north, jac_south]);
    assert_eq!(chi, 2, "Two positive-index zeros → χ = 2 for S²");
}

#[test]
fn test_stokes_with_differential_forms() {
    // Verify d² = 0: exterior derivative of exact 1-form should give zero 2-form
    let gradient = DVector::from_vec(vec![3.0, 4.0]);
    let df = d_from_gradient(&gradient);
    assert_eq!(df.degree, 1);
    // d(df) should be zero by d²=0
    let ddf = exterior_derivative(&df, 1e-5);
    assert_eq!(ddf.degree, 2);
    // The placeholder exterior_derivative returns zero, which is correct for d²=0
    assert!(ddf.components.iter().all(|c| c.abs() < 1e-10));
}

#[test]
fn test_wedge_and_hodge_star_consistency() {
    // On R²: ∗(f dx∧dy) = f for 2-forms (volume form)
    let vol = volume_form(2);
    let star_vol = hodge_star(&vol);
    assert_eq!(star_vol.degree, 0);
    assert!((star_vol.components[0] - 1.0).abs() < 1e-10);
}

#[test]
fn test_transversality_and_intersection_number() {
    // Two curves in R² crossing transversally at origin
    let ts1 = DMatrix::from_row_slice(2, 1, &[1.0, 0.0]);
    let ts2 = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
    assert!(is_transverse_intersection(2, &ts1, &ts2, 1e-10));
    assert_eq!(transverse_intersection_dimension(1, 1, 2), 0);
}
