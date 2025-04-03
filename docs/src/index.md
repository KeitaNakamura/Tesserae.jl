# Tesserae.jl

*A Julia-powered toolkit for material point method*

## Introduction

Tesserae is a toolkit for implementing the material point method in Julia. It is designed to maintain consistency between mathematical expressions and source code, enabling rapid development. Current features include:

* Grid and particle generation
* Various interpolation types, including B-splines, GIMP, CPDI, and MLS--MPM
* Convenient macros for transferring data between grid and particles
* Multithreading support
* Exporting VTK files for visualization
* Unified framework for FEM (experimental)
* GPU computation with CUDA and Metal (experimental)

## Installation

You can add Tesserae using Julia's package manager, by typing `]add Tesserae` in the Julia REPL:

```julia
pkg> add Tesserae
```
