# Getting started

!!! info
    This tutorial builds a two-disk MPM simulation step by step. A runnable script is shown at the end.

The simulation in this tutorial uses four core objects:

1. A `mesh` stores the background node positions.
2. A `grid` stores user-defined fields on that mesh, such as mass, momentum, force, and velocity.
3. `particles` store material state, such as position, velocity, volume, deformation gradient, and stress.
4. `weights` store the basis function values connecting each particle to nearby grid nodes.

At each time step, we update the weights, use [`@P2G`](@ref) for particle-to-grid transfer and grid-node calculations, and then use [`@G2P`](@ref) for grid-to-particle transfer and particle calculations.

## Simulation loop at a glance

The full simulation loop has this shape:

```julia
for each time step
    update basis function values
    @P2G ...      # particles -> grid, then grid nodes
    @G2P ...      # grid -> particles, then particles
end
```

## Grid and particle generation

### 1. Simulation constants

Start by importing Tesserae and defining the constants used later:

```@example stepbystep
using Tesserae
Δt = 0.001                  # Time step size
E  = 1000.0                 # Young's modulus
ν  = 0.3                    # Poisson's ratio
λ  = (E*ν) / ((1+ν)*(1-2ν)) # Lame's first parameter
μ  = E / 2(1 + ν)           # Shear modulus
ρ⁰ = 1000.0                 # Initial density
nothing #hide
```

### 2. Grid and particle properties

Before generating the grid and particles, define the fields stored on each grid node and particle:

```@example stepbystep
struct GridProp
    x  :: Vec{2, Float64} # Position
    m  :: Float64         # Mass
    mv :: Vec{2, Float64} # Momentum
    f  :: Vec{2, Float64} # Force
    v  :: Vec{2, Float64} # Velocity
    vⁿ :: Vec{2, Float64} # Velocity at t = tⁿ
end
struct ParticleProp
    x  :: Vec{2, Float64}                           # Position
    m  :: Float64                                   # Mass
    V⁰ :: Float64                                   # Initial volume
    v  :: Vec{2, Float64}                           # Velocity
    ∇v :: SecondOrderTensor{2, Float64, 4}          # Velocity gradient
    F  :: SecondOrderTensor{2, Float64, 4}          # Deformation gradient
    σ  :: SymmetricSecondOrderTensor{2, Float64, 3} # Cauchy stress
end
nothing #hide
```

These properties are customizable: add the variables you want to store on each grid node or particle, using field types such as numbers, `Vec`s, and tensors. The first field must be the position; these position values are automatically set during grid and particle generation.

!!! info
    The same property layout can also be written with `@NamedTuple`.
    An `@NamedTuple` definition can be placed inside a function as a local binding, while `struct` definitions are global. For this reason, tutorials in this manual generally use `@NamedTuple` for property definitions.

    ```@example namedtuple_props
    using Tesserae # hide
    GridProp = @NamedTuple begin
        x  :: Vec{2, Float64} # Position
        m  :: Float64         # Mass
        mv :: Vec{2, Float64} # Momentum
        f  :: Vec{2, Float64} # Force
        v  :: Vec{2, Float64} # Velocity
        vⁿ :: Vec{2, Float64} # Velocity at t = tⁿ
    end
    nothing #hide
    ```

!!! info
    The property type must satisfy `isbitstype`, so avoid fields such as `Vector`, `String`, `Dict`, or `Any` here.

    ```@example stepbystep
    (isbitstype(GridProp), isbitstype(ParticleProp))
    ```

### 3. Mesh and grid generation

A mesh stores only node positions. A grid attaches the user-defined fields from `GridProp` to that mesh.

Create a Cartesian mesh with `CartesianMesh(spacing, (xmin, xmax), (ymin, ymax)...)`:

```@example stepbystep
mesh = CartesianMesh(0.05, (0,1), (0,1))
```

To keep the plotting examples short, use this helper to show the background mesh and, when provided, points on top of it:

```@example stepbystep
import Plots

function plot_state(mesh, xs=nothing)
    xmin, ymin = mesh[begin]
    xmax, ymax = mesh[end]
    pad = 0.1 * max(xmax - xmin, ymax - ymin)
    plt = Plots.plot(
        xlims = (xmin - pad, xmax + pad),
        ylims = (ymin - pad, ymax + pad),
        aspect_ratio = :equal,
        label = false,
    )
    for i in axes(mesh, 1)
        Plots.plot!(plt, Tuple.(mesh[i, :]); color=:lightgray, linewidth=0.5, label=false)
    end
    for j in axes(mesh, 2)
        Plots.plot!(plt, Tuple.(mesh[:, j]); color=:lightgray, linewidth=0.5, label=false)
    end
    if !isnothing(xs)
        Plots.scatter!(plt, Tuple.(xs); markersize=3, label=false)
    end
    plt
end
nothing #hide
```

This shows the background mesh used for transfers:

```@example stepbystep
plot_state(mesh)
```

Generate the grid from this mesh with:

```@example stepbystep
grid = generate_grid(GridProp, mesh)
```

This `grid` is a [`StructArray`](https://github.com/JuliaArrays/StructArrays.jl) with an element type of `GridProp`. Each field defined in `GridProp` can be accessed with dot notation:

```@example stepbystep
grid.v
```

!!! info
    Since the first field of `GridProp` is named `x`, Tesserae treats it as the grid-node position field. For this grid, those positions are given by the mesh itself.

    ```@example stepbystep
    grid.x === mesh
    ```

### 4. Particle generation

Generate particles with `generate_particles`:

```@example stepbystep
pts = generate_particles(ParticleProp, mesh; alg=GridSampling())
```

The generated particles cover the full mesh:

```@example stepbystep
plot_state(mesh, pts.x)
```

Like `grid`, `pts` is also a [`StructArray`](https://github.com/JuliaArrays/StructArrays.jl).

Particles are generated over the entire `mesh` domain, so set their initial volume before removing the particles outside the two disks:

```@example stepbystep
pts.V⁰ .= volume(mesh) / length(pts)
nothing #hide
```

Then keep the left and right disks, and set their initial velocities:

```@example stepbystep
r = 0.2 # Radius
lhs = findall(x -> norm(@. x-r    ) < r, pts.x)
rhs = findall(x -> norm(@. x-(1-r)) < r, pts.x)

pts.v[lhs] .= Ref(Vec( 0.1, 0.1))
pts.v[rhs] .= Ref(Vec(-0.1,-0.1))

particles = pts[[lhs; rhs]]
nothing #hide
```

The remaining particles form the two disks:

```@example stepbystep
plot_state(mesh, particles.x)
```

Set the mass and initial deformation gradient on each particle:

```@example stepbystep
@. particles.m = ρ⁰ * particles.V⁰
@. particles.F = one(particles.F)
nothing #hide
```

## Basis function values

In Tesserae, basis function values are stored in a `BasisWeight`.
For a linear basis, construct one as follows:

```@example stepbystep
bw = BasisWeight(BSpline(Linear()), mesh)
```

Update it at a particle position with `update!`:

```@example stepbystep
update!(bw, particles.x[1], mesh)
```

!!! info
    After updating `bw`, you can check the partition of unity $\sum_i w_{ip} = 1$:
    ```@example stepbystep
    sum(bw.w) ≈ 1
    ```
    and the linear field reproduction $\sum_i w_{ip} \bm{x}_i = \bm{x}_p$:
    ```@example stepbystep
    nodeindices = supportnodes(bw)
    x = sum(eachindex(nodeindices)) do ip
        i = nodeindices[ip]
        bw.w[ip] * mesh[i]
    end
    x ≈ particles.x[1]
    ```

The `bw` above is useful for looking at one particle. In a simulation, however, the transfer step needs basis function values for all particles. Prepare one `BasisWeight` per particle so each particle has its own storage:

```@example stepbystep
weights = map(p -> BasisWeight(BSpline(Linear()), mesh), eachindex(particles))
nothing #hide
```

!!! info
    You can also construct `BasisWeight`s with structure-of-arrays (SoA) layout using `generate_basis_weights`.

    ```@example stepbystep
    soa_weights = generate_basis_weights(BSpline(Linear()), mesh, length(particles))
    ```
    This SoA layout is generally preferred for performance, although it cannot be resized.

## Transfer between grid and particles

### Particle-to-grid transfer

Before transferring quantities, update the basis function values at the current particle positions:

```@example stepbystep
for p in eachindex(particles)
    update!(weights[p], particles.x[p], mesh)
end
nothing #hide
```

!!! info
    The same update can also be written in one line. This form uses Tesserae's backend-aware implementation, so it works with CPU threading and GPU arrays.

    ```@example stepbystep
    update!(weights, particles, mesh)
    nothing #hide
    ```

Use [`@P2G`](@ref) for particle-to-grid transfer and grid-node calculations:

```@example stepbystep
@P2G grid=>i particles=>p weights=>ip begin
    # Particle-to-grid transfer
    m[i]  = @∑ w[ip] * m[p]
    mv[i] = @∑ w[ip] * m[p] * v[p]
    f[i]  = @∑ -V⁰[p] * det(F[p]) * σ[p] * ∇w[ip]

    # Grid-node calculation
    mᵢ⁻¹ = iszero(m[i]) ? zero(m[i]) : inv(m[i])
    vⁿ[i] = mv[i] * mᵢ⁻¹
    v[i]  = vⁿ[i] + (f[i] * mᵢ⁻¹) * Δt
end
```

Equations using `@∑` perform the particle-to-grid scatter. The following equations are grid-node calculations, executed after the scatter for each grid node.

After this block, only grid nodes near the particles have nonzero mass:

```@example stepbystep
active = findall(m -> !iszero(m), grid.m)
plot_state(mesh, mesh[active])
```

The [`@P2G`](@ref) macro expands to roughly the following code:

```julia
@. grid.m  = zero(grid.m)
@. grid.mv = zero(grid.mv)
@. grid.f  = zero(grid.f)

# Particle-to-grid transfer
for p in eachindex(particles)
    bw = weights[p]
    nodeindices = supportnodes(bw)
    for ip in eachindex(nodeindices)
        i = nodeindices[ip]
        grid.m[i]  += bw.w[ip] * particles.m[p]
        grid.mv[i] += bw.w[ip] * particles.m[p] * particles.v[p]
        grid.f[i]  += -particles.V⁰[p] * det(particles.F[p]) * particles.σ[p] * bw.∇w[ip]
    end
end

# Grid-node calculation
for i in eachindex(grid)
    mᵢ⁻¹ = iszero(grid.m[i]) ? zero(grid.m[i]) : inv(grid.m[i])
    grid.vⁿ[i] = grid.mv[i] * mᵢ⁻¹
    grid.v[i]  = grid.vⁿ[i] + (grid.f[i] * mᵢ⁻¹) * Δt
end
```

### Grid-to-particle transfer

Use [`@G2P`](@ref) for grid-to-particle transfer and particle calculations:

```@example stepbystep
@G2P grid=>i particles=>p weights=>ip begin
    # Grid-to-particle transfer
    v[p] += @∑ w[ip] * (v[i] - vⁿ[i])
    ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
    x[p] += @∑ w[ip] * v[i] * Δt

    # Particle calculation
    Δεₚ = symmetric(∇v[p]) * Δt
    F[p]  = (I + ∇v[p]*Δt) * F[p]
    σ[p] += λ*tr(Δεₚ)*I + 2μ*Δεₚ # Linear elastic material
end
```

Equations using `@∑` gather grid quantities to each particle. The following equations are particle calculations, executed after the gathered values have been stored.

This macro expands to roughly the following code:

```julia
# Grid-to-particle transfer
for p in eachindex(particles)
    bw = weights[p]
    nodeindices = supportnodes(bw)
    Δvₚ = zero(eltype(particles.v))
    ∇vₚ = zero(eltype(particles.∇v))
    Δxₚ = zero(eltype(particles.x))
    for ip in eachindex(nodeindices)
        i = nodeindices[ip]
        Δvₚ += bw.w[ip] * (grid.v[i] - grid.vⁿ[i])
        ∇vₚ += grid.v[i] ⊗ bw.∇w[ip]
        Δxₚ += bw.w[ip] * grid.v[i] * Δt
    end
    particles.v[p] += Δvₚ
    particles.∇v[p] = ∇vₚ
    particles.x[p] += Δxₚ

    # Particle calculation
    Δεₚ = symmetric(particles.∇v[p]) * Δt
    particles.F[p]  = (I + particles.∇v[p]*Δt) * particles.F[p]
    particles.σ[p] += λ*tr(Δεₚ)*I + 2μ*Δεₚ # Linear elastic material
end
```

## Complete script

Putting the pieces together gives the following runnable script:

```@example complete_getting_started
using Tesserae
import Plots

function main()

    # Material constants and time step
    Δt = 0.001                  # Time step size
    E  = 1000.0                 # Young's modulus
    ν  = 0.3                    # Poisson's ratio
    λ  = (E*ν) / ((1+ν)*(1-2ν)) # Lame's first parameter
    μ  = E / 2(1 + ν)           # Shear modulus
    ρ⁰ = 1000.0                 # Initial density

    # Grid and particle properties
    GridProp = @NamedTuple begin
        x  :: Vec{2, Float64} # Position
        m  :: Float64         # Mass
        mv :: Vec{2, Float64} # Momentum
        f  :: Vec{2, Float64} # Force
        v  :: Vec{2, Float64} # Velocity
        vⁿ :: Vec{2, Float64} # Velocity at t = tⁿ
    end
    ParticleProp = @NamedTuple begin
        x  :: Vec{2, Float64}                           # Position
        m  :: Float64                                   # Mass
        V⁰ :: Float64                                   # Initial volume
        v  :: Vec{2, Float64}                           # Velocity
        ∇v :: SecondOrderTensor{2, Float64, 4}          # Velocity gradient
        F  :: SecondOrderTensor{2, Float64, 4}          # Deformation gradient
        σ  :: SymmetricSecondOrderTensor{2, Float64, 3} # Cauchy stress
    end

    # Mesh
    mesh = CartesianMesh(0.05, (0,1), (0,1))

    # Background grid
    grid = generate_grid(GridProp, mesh)

    # Particles
    particles = let
        pts = generate_particles(ParticleProp, mesh; alg=GridSampling())
        pts.V⁰ .= volume(mesh) / length(pts) # Set initial volume

        # Left and right disks
        r = 0.2 # Radius
        lhs = findall(x -> norm(@. x-r    ) < r, pts.x)
        rhs = findall(x -> norm(@. x-(1-r)) < r, pts.x)

        # Set initial velocities
        pts.v[lhs] .= Ref(Vec( 0.1, 0.1))
        pts.v[rhs] .= Ref(Vec(-0.1,-0.1))

        pts[[lhs; rhs]]
    end
    @. particles.m = ρ⁰ * particles.V⁰
    @. particles.F = one(particles.F)

    # Basis weights
    weights = generate_basis_weights(BSpline(Linear()), mesh, length(particles))

    # Create animation with `Plots.@gif`
    Plots.@gif for t in range(0, 4, step=Δt)

        # Update basis function values
        update!(weights, particles, mesh)

        @P2G grid=>i particles=>p weights=>ip begin
            # Particle-to-grid transfer
            m[i]  = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
            f[i]  = @∑ -V⁰[p] * det(F[p]) * σ[p] * ∇w[ip]

            # Grid-node calculation
            mᵢ⁻¹ = iszero(m[i]) ? zero(m[i]) : inv(m[i])
            vⁿ[i] = mv[i] * mᵢ⁻¹
            v[i]  = vⁿ[i] + (f[i] * mᵢ⁻¹) * Δt
        end

        @G2P grid=>i particles=>p weights=>ip begin
            # Grid-to-particle transfer
            v[p] += @∑ w[ip] * (v[i] - vⁿ[i])
            ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
            x[p] += @∑ w[ip] * v[i] * Δt

            # Particle calculation
            Δεₚ = symmetric(∇v[p]) * Δt
            F[p]  = (I + ∇v[p]*Δt) * F[p]
            σ[p] += λ*tr(Δεₚ)*I + 2μ*Δεₚ # Linear elastic material
        end

        # Plot the current particle positions
        plot_state(mesh, particles.x)
    end every 100

end

function plot_state(mesh, xs=nothing)
    xmin, ymin = mesh[begin]
    xmax, ymax = mesh[end]
    pad = 0.1 * max(xmax - xmin, ymax - ymin)
    plt = Plots.plot(
        xlims = (xmin - pad, xmax + pad),
        ylims = (ymin - pad, ymax + pad),
        aspect_ratio = :equal,
        label = false,
    )
    for i in axes(mesh, 1)
        Plots.plot!(plt, Tuple.(mesh[i, :]); color=:lightgray, linewidth=0.5, label=false)
    end
    for j in axes(mesh, 2)
        Plots.plot!(plt, Tuple.(mesh[:, j]); color=:lightgray, linewidth=0.5, label=false)
    end
    if !isnothing(xs)
        Plots.scatter!(plt, Tuple.(xs); markersize=3, label=false)
    end
    plt
end

main()
```
