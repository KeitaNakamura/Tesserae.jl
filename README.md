# Tesserae.jl

*A Julia-powered toolkit for material point method*

[![CI](https://github.com/KeitaNakamura/Tesserae.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/KeitaNakamura/Tesserae.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/KeitaNakamura/Tesserae.jl/graph/badge.svg?token=H5BHWIIBTG)](https://codecov.io/gh/KeitaNakamura/Tesserae.jl)

Tesserae is a toolkit for implementing the material point method in Julia. It is designed to maintain consistency between mathematical expressions and source code, enabling rapid development. Current features include:

* Grid and particle generation
* Various interpolation types, including B-splines, GIMP, CPDI, and MLS--MPM
* Convenient macros for transferring data between grids and particles
* Multithreading support
* Exporting VTK files for visualization

## Documentation

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://KeitaNakamura.github.io/Tesserae.jl/stable)
[![Develop](https://img.shields.io/badge/docs-dev-blue.svg)](https://KeitaNakamura.github.io/Tesserae.jl/dev)

## Examples

The following examples are taken from the tutorials.

### Various velocity transfer schemes (PIC, FLIP, APIC...)

<img src="https://github.com/user-attachments/assets/a069b594-389f-4082-8755-3d8b90908c67" width="500"/>

### Jacobian-free Newton-Krylov method

<img src="https://github.com/user-attachments/assets/f1d80c46-a8ff-44d4-ae82-768b480f25ea" width="500"/>

### Stabilized mixed MPM for incompressible fluid flow

<img src="https://github.com/user-attachments/assets/76fd800e-fda7-4d89-afcd-9a8a2178ab41" width="500"/>

## Other MPM packages in Julia

* [MaterialPointSolver.jl](https://github.com/LandslideSIM/MaterialPointSolver.jl)

## Inspiration

Some functionalities are inspired from the following packages:

* [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl)
* [taichi_mpm](https://github.com/yuanming-hu/taichi_mpm)
