# Tesserae.jl

*A Julia-powered toolkit for material point method*

[![CI](https://github.com/KeitaNakamura/Tesserae.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/KeitaNakamura/Tesserae.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/KeitaNakamura/Tesserae.jl/graph/badge.svg?token=H5BHWIIBTG)](https://codecov.io/gh/KeitaNakamura/Tesserae.jl)

Tesserae is a toolkit for implementing the material point method in Julia. It is designed to maintain consistency between mathematical expressions and source code, enabling rapid development. Current features include:

* Grid and particle generation
* Various interpolation types, including B-splines, GIMP, CPDI, and MLS-MPM
* Convenient macros for transferring data between grid and particles
* Multi-threading support
* GPU support (CUDA and Metal)
* Exporting VTK files for visualization
* Unified framework for FEM (experimental)

![Image](https://github.com/user-attachments/assets/62acb8c4-3705-4202-b30c-fff64a690479)
![Image](https://github.com/user-attachments/assets/16379e40-9a5c-404c-9543-251e210fb00e)
![Image](https://github.com/user-attachments/assets/00bf6683-010d-4093-9cda-a5f772fe5a74)
![Image](https://github.com/user-attachments/assets/4fa4dfa2-f1cb-471e-99f1-0aa0783a2f20)

## Documentation

[![Stable](https://img.shields.io/badge/docs-latest%20release-blue.svg)](https://KeitaNakamura.github.io/Tesserae.jl/stable)

## Other MPM packages in Julia

* [MaterialPointSolver.jl](https://github.com/LandslideSIM/MaterialPointSolver.jl)

## Inspiration

Some functionalities are inspired from the following packages:

* [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl)
* [Flux.jl](https://github.com/FluxML/Flux.jl)
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
