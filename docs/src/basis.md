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

This trades arithmetic for memory traffic. Deferring pays the evaluation on every
transfer, while storing pays it once per `update!` and amortizes it over the
transfers that follow, so the comparison depends on how many transfers one
`update!` serves:

- **On the CPU, storing wins.** Deferring is already about 1.8x slower at one
  transfer per `update!` and falls further behind as that count grows. Its value
  on the CPU is memory: nothing is allocated for the values.
- **On the GPU it depends on the basis.** For [`BSpline`](@ref) the recomputation
  costs less than the memory traffic it replaces, so deferring is faster at every
  transfer count. For [`uGIMP`](@ref), evaluation is dearer relative to its
  footprint and deferring only wins below a few matvecs per `update!`.

The memory saved is substantial for high-order bases in 3D: storing a cubic
B-spline's values and gradients for a million particles takes about 1 GB, which
deferring removes entirely.

!!! warning "When the values are taken"
    Stored values date from the last [`update!`](@ref); deferred values are
    evaluated by the transfer that reads them, from the particle state as it
    stands when that transfer begins. Within one transfer the two agree, and a
    [`@G2P2G`](@ref) that writes `x[p]` between its halves still evaluates both
    halves at the state it started from. Across transfers they can differ: if
    particles move between two transfers and no `update!` runs in between,
    stored weights keep the old positions while deferred weights follow the new
    ones. Call `update!` after moving particles — which a loop written for
    stored weights already does — and the two agree.

A deferred basis must be able to produce one node's value from the particle and
the mesh alone. [`BSpline`](@ref), [`SteffenBSpline`](@ref) and [`uGIMP`](@ref)
qualify. [`WLS`](@ref) and [`KernelCorrection`](@ref) do not, because each node's
value depends on a fit over the whole support, and they are rejected with an
error at construction. The support window itself is always derived on the fly,
so it is not stored either.

### Choosing per step

When the number of transfers per `update!` varies from step to step — as in an
implicit solve, where a matrix-free stiffness application is a
[`@G2P`](@ref)/[`@P2G`](@ref) pair and the count follows the convergence of the
linear solver — neither choice is right for every step.
[`Tesserae.deferred`](@ref) gives a deferred view of stored weights, so both are
available at once and each step can pass whichever it wants. The view shares its
support nodes with the stored weights and owns no storage, so making one is free
and the stored values stay intact and reusable.

```@docs
Tesserae.deferred
Tesserae.is_lazy
```
