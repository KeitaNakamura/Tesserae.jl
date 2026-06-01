```@meta
CollapsedDocStrings = false
```

# Basis Functions

Basis weights store the values and gradients of a basis function between particles and nearby grid nodes.
They are updated after particles move and then reused by transfer macros such as [`@P2G`](@ref) and [`@G2P`](@ref).

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
