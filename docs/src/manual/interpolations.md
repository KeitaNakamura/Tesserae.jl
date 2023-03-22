```@meta
CurrentModule = Marble
```

# Interpolations

## Interpolation types

### `Kernel`s

In Marble.jl, `Kernel`s are defined as core weight functions.
The available kernels are follows:

* [`LinearBSpline()`](@ref)
* [`QuadraticBSpline()`](@ref)
* [`CubicBSpline()`](@ref)
* [`uGIMP()`](@ref)

```@docs
LinearBSpline
QuadraticBSpline
CubicBSpline
uGIMP
```

### `Interpolation`s

`Interpolation` is a `supertype` of `Kernel`. This type is not fundamentally different from `Kernel`,
but it is distinguished from `Kernel` because some interpolation methods needs weight functions (`Kernel`s).
For example, MLS-MPM (moving least squares MPM) uses B-spline functions as weight functions to build up the MLS shape function.
Marble.jl supports following advanced interpolation methods:

* [`LinearWLS(::Kernel)`](@ref)
* [`KernelCorrection(::Kernel)`](@ref)

These interpolations can not be used only, and always needs kernels.

```@docs
LinearWLS
KernelCorrection
```

## Interpolation space (`MPSpace`)

```@docs
MPSpace
update!(::MPSpace, ::Grid, ::Particles)
```
