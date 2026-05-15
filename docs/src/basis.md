```@meta
CollapsedDocStrings = false
```

# Basis Functions

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
