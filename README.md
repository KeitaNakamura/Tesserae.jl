# Tesserae.jl

*A Julia-powered toolkit for material point method*

[![CI](https://github.com/KeitaNakamura/Tesserae.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/KeitaNakamura/Tesserae.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/KeitaNakamura/Tesserae.jl/graph/badge.svg?token=H5BHWIIBTG)](https://codecov.io/gh/KeitaNakamura/Tesserae.jl)

Tesserae is a toolkit for implementing the material point method in Julia. It is designed to maintain consistency between mathematical expressions and source code, enabling rapid development. Current features include:

* Grid and particle generation
* Various interpolation types, including B-splines, GIMP, CPDI, and MLS-MPM
* Convenient macros for transferring data between grid and particles
* Multithreading support
* Exporting VTK files for visualization
* Unified framework for FEM (experimental)
* GPU computation with CUDA and Metal (experimental)

## Documentation

[![Stable](https://img.shields.io/badge/docs-latest%20release-blue.svg)](https://KeitaNakamura.github.io/Tesserae.jl/stable)

## Examples

The following examples are taken from the tutorials.

### Simulations using various transfers (PIC, FLIP, APIC, TPIC, and XPIC) [🔗](https://keitanakamura.github.io/Tesserae.jl/stable/tutorials/collision/)

<img src="https://github.com/user-attachments/assets/24cbd50c-7d21-4917-a7e1-12816b561dee" width="500"/>

### Jacobian-free Newton-Krylov method [🔗](https://keitanakamura.github.io/Tesserae.jl/stable/tutorials/implicit_jacobian_free/)

<img src="https://github.com/user-attachments/assets/9d9dbb86-87d5-4818-bbbf-ae0983cd3f04" width="500"/>

### Stabilized mixed MPM for incompressible fluid flow [🔗](https://keitanakamura.github.io/Tesserae.jl/stable/tutorials/dam_break/)

<img src="https://github.com/user-attachments/assets/dfc9fb4e-6223-460e-ac34-310363cd6a78" width="500"/>

## Other MPM packages in Julia

* [MaterialPointSolver.jl](https://github.com/LandslideSIM/MaterialPointSolver.jl)

## Inspiration

Some functionalities are inspired from the following packages:

* [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl)
* [taichi_mpm](https://github.com/yuanming-hu/taichi_mpm)

## Citation

If you find Tesserae.jl useful in your work, I kindly request that you cite it as below:

```bibtex
@software{NakamuraTesserae2024,
    title = {Tesserae.jl: a {J}ulia-powered toolkit for material point method},
   author = {Nakamura, Keita},
      doi = {10.5281/zenodo.13956709},
     year = {2024},
      url = {https://github.com/KeitaNakamura/Tesserae.jl}
  licence = {MIT},
}
```
