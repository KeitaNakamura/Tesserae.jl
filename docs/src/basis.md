```@meta
CollapsedDocStrings = false
```

# Basis Functions

Basis functions define the interpolation between particles and grid nodes.
Tesserae stores this local particle-grid relation as basis weights, which contain the basis values and gradients used by transfer macros such as [`@P2G`](@ref) and [`@G2P`](@ref).

Because particles move through the mesh, basis weights are updated before transfers that use the current particle positions.
The basis type determines the support nodes of each particle, affecting both the transfer behavior and computational cost.

```@docs
update!(::AbstractArray{<: BasisWeight}, ::Tesserae.StructArray, ::Tesserae.AbstractMesh, ::AbstractArray)
```

## Basis types

```@docs
Basis
BSpline
SteffenBSpline
uGIMP
CPDI
WLS
KernelCorrection
```

## Basis weight

```@docs
BasisWeight
BasisWeightArray
generate_basis_weights
basis
supportnodes
```
