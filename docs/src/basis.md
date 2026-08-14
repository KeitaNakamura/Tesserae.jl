```@meta
CollapsedDocStrings = false
```

# Basis Functions

Basis functions define the interpolation between particles and grid nodes.
Tesserae stores this local particle-grid relation as basis weights, which contain the basis values and gradients used by transfer macros such as [`@P2G`](@ref) and [`@G2P`](@ref).

Because particles move through the mesh, basis weights are updated before transfers that use the current particle positions.
The basis type determines the support nodes of each particle, affecting both the transfer behavior and computational cost.

```@docs
update!(::AbstractArray{<: BasisWeight}, ::Tesserae.StructArray, ::Tesserae.AbstractMesh, ::AbstractArray{Bool})
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

## Deferred basis weights

By default the basis values are computed by [`update!`](@ref) and stored, so each
transfer reads them back. Passing `lazy=true` to [`generate_basis_weights`](@ref)
allocates no storage for them and evaluates them inside each transfer instead:

```julia
weights = generate_basis_weights(BSpline(Quadratic()), mesh, length(particles); lazy=true)
update!(weights, particles, mesh)  # no-op, so an existing loop needs no change
```

This trades arithmetic for memory traffic, and which side wins is a property of
the backend rather than of the step, so the choice is made at construction and
cannot be switched afterwards:

- **On the GPU it is faster**, because the values it recomputes cost less than
  the memory traffic of reading them back. Deferred weights are the better
  default for GPU transfers.
- **On the CPU it is slower** (roughly 1.8x even at one transfer per `update!`,
  and worse as the number of transfers per update grows, since `update!`
  amortizes the same work over all of them). Its value there is memory: nothing
  is allocated for the values, which is a few hundred megabytes at a few million
  particles.

A deferred basis must be able to produce one node's value from the particle and
the mesh alone. [`BSpline`](@ref), [`SteffenBSpline`](@ref) and [`uGIMP`](@ref)
qualify. [`WLS`](@ref) and [`KernelCorrection`](@ref) do not, because each node's
value depends on a fit over the whole support, and they are rejected with an
error at construction. The support window itself is always derived on the fly,
so it is not stored either.

```@docs
Tesserae.is_lazy
```
