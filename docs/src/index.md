# Tesserae.jl

*A Julia toolkit for the material point method*

## Introduction

Tesserae is a toolkit for implementing the material point method in Julia. It is designed to maintain consistency between mathematical expressions and source code, enabling rapid development. Current features include:

* Grid and particle generation
* Various basis functions, including B-splines, GIMP, CPDI, and MLS--MPM
* Convenient macros for transferring data between grid and particles
* Multi-threading support
* GPU support (CUDA and Metal)
* Exporting VTK files for visualization
* Unified framework for FEM and IGA (IGA support is experimental)

## Why Tesserae.jl?

Tesserae is intended for users who want to implement MPM algorithms directly rather than operate a black-box solver. The core objects are deliberately small: a background mesh, grid fields, particles, and basis weights. Transfer macros such as [`@P2G`](@ref) and [`@G2P`](@ref) keep the implementation close to the equations, while the same code structure can be moved from CPU execution to multi-threaded or GPU execution.

Tensors are represented as tensors, using [Tensorial.jl](https://github.com/KeitaNakamura/Tensorial.jl), so constitutive models and tangent operators can be written in notation that resembles the mathematical formulation.

## Where to start

If you are new to Tesserae, start with the [Getting started](@ref) tutorial. It introduces the basic workflow:

1. Define grid and particle properties.
2. Generate a mesh, grid, particles, and basis weights.
3. Transfer particle data to the grid with [`@P2G`](@ref).
4. Update grid quantities.
5. Transfer grid data back to particles with [`@G2P`](@ref).
6. Update particle state and write or plot results.

After that, the [Transfer schemes](@ref) tutorial is a good next step for comparing FLIP, APIC, TPIC, and XPIC.
The [Manual overview](@ref manual_overview) connects the tutorial workflow to the individual API pages.
For larger simulations, see [Multi-threading](@ref) and [GPU computing](@ref).

## Installation

You can add Tesserae using Julia's package manager, by typing `]add Tesserae` in the Julia REPL:

```julia
pkg> add Tesserae
```
