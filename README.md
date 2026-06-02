# difftop-rs

Differential topology in Rust. Smooth manifolds, forms, and Stokes' theorem.

A library for **differential topology**: smooth manifolds, tangent bundles, differential forms, Stokes' theorem, transversality, degree theory, and the Euler characteristic via Poincaré–Hopf.

Built on `nalgebra` for linear algebra and `serde` for serialization.

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
difftop-rs = "0.1"
```

Requires Rust 2021 edition.

## Quick Start

### Build a manifold and inspect its atlas

```rust
use difftop_rs::sphere_manifold_2d;

let s2 = sphere_manifold_2d(); // S² with two stereographic charts
assert_eq!(s2.dimension(), 2);
assert_eq!(s2.atlas.charts.len(), 2);
```

### Define a vector field, find equilibria, classify them

```rust
use difftop_rs::{VectorField, classify_equilibrium};
use nalgebra::DVector;

let vf = VectorField::new("saddle", 2, |p: &DVector<f64>| {
    DVector::from_vec(vec![p[0], -p[1]])
});

let eqs = vf.find_equilibria_grid(&[(-2.0, 2.0), (-2.0, 2.0)], 30);
for eq in &eqs {
    let jac = /* numerical jacobian */;
    println!("{:?} at {:?}", classify_equilibrium(&jac), eq);
}
```

### Compute winding number and Brouwer degree

```rust
use difftop_rs::winding_number;
use nalgebra::DVector;

let wn = winding_number(
    &|t| DVector::from_vec(vec![t.cos(), t.sin()]),
    &|t| DVector::from_vec(vec![-t.sin(), t.cos()]),
    &DVector::from_vec(vec![0.0, 0.0]),
    1000,
);
assert!((wn - 1.0).abs() < 0.01);
```

### Euler characteristic via Poincaré–Hopf

```rust
use difftop_rs::{euler_characteristic_poincare_hopf, euler_characteristic_sphere};
use nalgebra::DMatrix;

let chi = euler_characteristic_poincare_hopf(&[
    DMatrix::identity(2, 2),
    DMatrix::identity(2, 2),
]);
assert_eq!(chi, 2);
assert_eq!(euler_characteristic_sphere(2), 2);
```

## Modules

| Module | What you get |
|---|---|
| `smooth_manifold` | Charts, atlases, transition maps, pre-built manifolds (Sⁿ, T², Rⁿ) |
| `tangent` | Tangent vectors, tangent/cotangent spaces, tangent bundles, pushforwards |
| `vector_field` | Smooth vector fields, integral curves (RK4), equilibrium classification |
| `differential_form` | k-forms, wedge product, exterior derivative, pullback, Hodge star |
| `stokes` | Integration of forms, line/surface integrals, Stokes' theorem verification |
| `transversality` | Transverse intersection checking, intersection dimension |
| `degree` | Brouwer degree, winding number, homotopy invariance |
| `euler` | Euler characteristic (Poincaré–Hopf, triangulation, Morse theory, Gauss–Bonnet) |
| `agent_dynamics` | Multi-agent systems on manifolds, flow analysis, Lyapunov exponents |

## License

MIT OR Apache-2.0
