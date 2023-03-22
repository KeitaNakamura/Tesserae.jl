```@meta
CurrentModule = Marble
```

# Transfers

This page explains MPM transfers between grid and particles.

## Transfer algorithms

* [Basic transfers](@ref)
* [Affine transfers](@ref)
* [Taylor transfers](@ref)

```@docs
TransferAlgorithm
```

### Basic transfers

* [`FLIP`](@ref)
* [`PIC`](@ref)

```@docs
FLIP
PIC
```

### Affine transfers

```@docs
AffineTransfer
AFLIP
APIC
```

### Taylor transfers

```@docs
TaylorTransfer
TFLIP
TPIC
```

## Transfer functions

* [`particle_to_grid!`](@ref)
* [`grid_to_particle!`](@ref)

```@docs
particle_to_grid!
grid_to_particle!
```
