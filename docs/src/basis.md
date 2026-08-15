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

By default [`update!`](@ref) computes the basis values and stores them, and each
transfer reads them back. Deferred weights store nothing and evaluate the values
inside the transfer that needs them.

Build them by passing `deferred=true` to [`generate_basis_weights`](@ref):

```julia
weights = generate_basis_weights(BSpline(Quadratic()), mesh, length(particles); deferred=true)
```

`update!` on such weights has nothing to fill and does nothing, so a loop written
for stored weights runs unchanged.

Stored weights can also be switched over for a single step, which is useful when
the number of transfers per `update!` varies — as in an implicit solve, where a
matrix-free stiffness application is a [`@G2P`](@ref)/[`@P2G`](@ref) pair and the
count follows the convergence of the linear solver:

```julia
update!(weights, particles, mesh; deferred=true)   # evaluate inside the transfers
update!(weights, particles, mesh)                  # fill the values and read them
```

The keyword leaves the stored values in place and only stops transfers from
reading them, so switching back costs a refill and nothing else. The transfers
themselves are written the same way either way.

### Which to use

Deferring trades memory traffic for arithmetic: it saves the storage and the
reads, and pays the evaluation again on every transfer, where storing pays it
once per `update!` and amortizes it over the transfers that follow.

Which wins depends on the backend, the basis and how many transfers one `update!`
serves, so it is worth measuring. As a starting point, deferring tends to pay off
on a GPU, where the values it recomputes can cost less than the memory traffic
they replace, and to lose on a CPU, where reading is cheap. Its other benefit is
independent of speed: high-order bases in 3D spend a large amount of memory on
stored values and gradients, and deferring removes that entirely.

### Requirements

A basis can defer if one node's value follows from the particle, the mesh, and at
most a quantity shared by the whole support. [`BSpline`](@ref),
[`SteffenBSpline`](@ref) and [`uGIMP`](@ref) need only the first two.
[`WLS`](@ref) and [`KernelCorrection`](@ref) fit their values over the support, so
the fit is computed once per particle before the support loop and each node reads
it. The support window is always derived on the fly, so it is not stored either.

`WLS` and `KernelCorrection` also consult the boundary filter passed to
`update!` to decide where to correct. `update!` records it, so a deferred
transfer corrects against the same nodes a stored one would, with no change at
the transfer. Bases outside the list above are refused rather than silently doing
nothing.

!!! warning "When the values are taken"
    Stored values date from the last [`update!`](@ref). Deferred values are taken
    from the particle state as it stands when a transfer begins, and a
    [`@G2P2G`](@ref) that writes `x[p]` between its halves still evaluates both
    halves at the state it started from.

    The two therefore differ if particles move between transfers with no
    `update!` in between: stored weights keep the old positions, deferred weights
    follow the new ones. Calling `update!` after moving particles, as a loop
    written for stored weights already does, keeps them in agreement.

```@docs
Tesserae.isdeferred
```
