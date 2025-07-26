# Getting started

!!! info
    Step-by-step instructions are provided after the code.

```@example
using Tesserae
import Plots

function main()

    # Material constants
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

    # Interpolation weights
    weights = map(p -> InterpolationWeight(BSpline(Linear()), mesh), eachindex(particles))

    # Create animation by `Plots.@gif`
    Δt = 0.001
    Plots.@gif for t in range(0, 4, step=Δt)

        # Update basis function values
        for p in eachindex(particles)
            update!(weights[p], particles.x[p], mesh)
        end

        @P2G grid=>i particles=>p weights=>ip begin
            m[i]  = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
            f[i]  = @∑ -V⁰[p] * det(F[p]) * σ[p] * ∇w[ip]
        end

        @. grid.vⁿ = grid.mv / grid.m
        @. grid.v  = grid.vⁿ + (grid.f / grid.m) * Δt

        @G2P grid=>i particles=>p weights=>ip begin
            v[p] += @∑ w[ip] * (v[i] - vⁿ[i])
            ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
            x[p] += @∑ w[ip] * v[i] * Δt
        end

        for p in eachindex(particles)
            Δϵₚ = symmetric(particles.∇v[p]) * Δt
            particles.F[p]  = (I + particles.∇v[p]*Δt) * particles.F[p]
            particles.σ[p] += λ*tr(Δϵₚ)*I + 2μ*Δϵₚ # Linear elastic material
        end

        # Plot results
        Plots.scatter(
            reinterpret(Tuple{Float64,Float64}, particles.x),
            lims = (0,1),
            ticks = 0:0.2:1,
            minorgrid = true,
            minorticks = 4,
            aspect_ratio = :equal,
            legend = false,
        )
    end every 100

end

main()
```

## Grid and particle generation

### 1. Grid and particle properties

```@example stepbystep
E  = 500                    #hide
ν  = 0.3                    #hide
λ  = (E*ν) / ((1+ν)*(1-2ν)) #hide
μ  = E / 2(1 + ν)           #hide
ρ⁰ = 1000                   #hide
nothing                     #hide
```

Before generating grid and particles, we must define their properties. This can be done by simply defining a `NamedTuple` for the grid and particles, respectively, as follows:

```@example stepbystep
using Tesserae # hide
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
nothing #hide
```

These properties are fully customizable, allowing users to define any variables. However, two conditions must be met: (1) the property type must be of `isbitstype`, and (2) the first variable must represent the position of the grid nodes and particles. These position values are automatically set during the grid and particle generation process.

```@repl stepbystep
isbitstype(GridProp)
isbitstype(ParticleProp)
```


!!! info
    The `struct` can also be used instead of `NamedTuple` as follows:
    ```@example
    using Tesserae # hide
    struct GridProp
        x  :: Vec{2, Float64} # Position
        m  :: Float64         # Mass
        mv :: Vec{2, Float64} # Momentum
        f  :: Vec{2, Float64} # Force
        v  :: Vec{2, Float64} # Velocity
        vⁿ :: Vec{2, Float64} # Velocity at t = tⁿ
    end
    ```

### 2. Mesh and grid generation

In Tesserae, a mesh and a grid differ in that a mesh only contains information about the positions of the nodes, whereas a grid includes the mesh (i.e., nodal positions) and additional user-defined variables (as defined in `GridProp` above).

To create Cartesian mesh, use `CartesianMesh(spacing, (xmin, xmax), (ymin, ymax)...)` as

```@example stepbystep
mesh = CartesianMesh(0.05, (0,1), (0,1))
```

Using this `mesh`, the grid can be generated as

```@example stepbystep
grid = generate_grid(GridProp, mesh)
```

This `grid` is a [`StructArray`](https://github.com/JuliaArrays/StructArrays.jl) with an element type of `GridProp`. Thus, each variable defined in `GridProp` can be accessed using the `.` notation as follows:

```@repl stepbystep
grid.v
```

!!! info
    Note that `grid.x` simply returns the mesh.
    ```@repl stepbystep
    grid.x === mesh
    ```

### 3. Particle generation

The particles can be generated using `generate_particles` function:

```@example stepbystep
pts = generate_particles(ParticleProp, mesh; alg=GridSampling())
```

This `pts` is also a [`StructArray`](https://github.com/JuliaArrays/StructArrays.jl), similar to `grid`.

!!! info
    In the `generate_particles` function, particles are generated across the entire `mesh` domain. Consequently, any unnecessary particles need to be removed.

```@example stepbystep
particles = let                                                      #hide
    pts = generate_particles(ParticleProp, mesh; alg=GridSampling()) #hide
    pts.V⁰ .= volume(mesh) / length(pts)                             #hide
    r = 0.2                                                          #hide
    lhs = findall(x -> norm(@. x-r    ) < r, pts.x)                  #hide
    rhs = findall(x -> norm(@. x-(1-r)) < r, pts.x)                  #hide
    pts.v[lhs] .= Ref(Vec( 0.1, 0.1))                                #hide
    pts.v[rhs] .= Ref(Vec(-0.1,-0.1))                                #hide
    pts[[lhs; rhs]]                                                  #hide
end                                                                  #hide
nothing                                                              #hide
```

## Basis function values

In Tesserae, the basis function values are stored in `InterpolationWeight`.
For example, `InterpolationWeight` with the linear basis function can be constructed as

```@repl stepbystep
iw = InterpolationWeight(BSpline(Linear()), mesh)
```

This `iw` can be updated by passing the particle position to the `update!` function:

```@repl stepbystep
update!(iw, particles.x[1], mesh)
```

!!! info
    After updating `iw`, you can check the partition of unity $\sum_i w_{ip} = 1$:
    ```@repl stepbystep
    sum(iw.w)
    ```
    and the linear field reproduction $\sum_i w_{ip} \bm{x}_i = \bm{x}_p$:
    ```@repl stepbystep
    nodeindices = neighboringnodes(iw)
    sum(eachindex(nodeindices)) do ip
        i = nodeindices[ip]
        iw.w[ip] * mesh[i]
    end
    ```

For the sake of performance, it's best to prepare the same number of `InterpolationWeight`s as there are particles. This means that each particle has its own storage for the basis function values.

```@example stepbystep
weights = map(p -> InterpolationWeight(BSpline(Linear()), mesh), eachindex(particles))
nothing #hide
```

!!! info
    It is also possible to construct `InterpolationWeight`s with Structure-Of-Arrays (SOA) layout using `generate_interpolation_weights`.
    ```@repl stepbystep
    weights = generate_interpolation_weights(BSpline(Linear()), mesh, length(particles))
    ```
    This SoA layout for `InterpolationWeight`s is generally preferred for performance, although it cannot be resized.

## Transfer between grid and particles

### Particle-to-grid transfer

```@example stepbystep
Δt = 0.001                                     #hide
for p in eachindex(particles)                  #hide
    update!(weights[p], particles.x[p], mesh) #hide
end                                            #hide
```

For the particle-to-grid transfer, the [`@P2G`](@ref) macro is useful:

```@example stepbystep
@P2G grid=>i particles=>p weights=>ip begin
    m[i]  = @∑ w[ip] * m[p]
    mv[i] = @∑ w[ip] * m[p] * v[p]
    f[i]  = @∑ -V⁰[p] * det(F[p]) * σ[p] * ∇w[ip]
end
```

This macro expands to roughly the following code:

```julia
@. grid.m  = zero(grid.m)
@. grid.mv = zero(grid.mv)
@. grid.f  = zero(grid.f)
for p in eachindex(particles)
    iw = weights[p]
    nodeindices = neighboringnodes(iw)
    for ip in eachindex(nodeindices)
        i = nodeindices[ip]
        grid.m[i]  += iw.w[ip] * particles.m[p]
        grid.mv[i] += iw.w[ip] * particles.m[p] * particles.v[p]
        grid.f[i]  += -particles.V⁰[p] * det(particles.F[p]) * particles.σ[p] * iw.∇w[ip]
    end
end
```

### Grid-to-particle transfer

Similar to the particle-to-grid transfer, the [`@G2P`](@ref) macro is provided for grid-to-particle transfer:

```@example stepbystep
@G2P grid=>i particles=>p weights=>ip begin
    v[p] += @∑ w[ip] * (v[i] - vⁿ[i])
    ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
    x[p] += @∑ w[ip] * v[i] * Δt
end
```

This macro expands to roughly the following code:

```julia
for p in eachindex(particles)
    iw = weights[p]
    nodeindices = neighboringnodes(iw)
    Δvₚ = zero(eltype(particles.v))
    ∇vₚ = zero(eltype(particles.∇v))
    Δxₚ = zero(eltype(particles.x))
    for ip in eachindex(nodeindices)
        i = nodeindices[ip]
        Δvₚ += iw.w[ip] * (grid.v[i] - grid.vⁿ[i])
        ∇vₚ += grid.v[i] ⊗ iw.∇w[ip]
        Δxₚ += iw.w[ip] * grid.v[i] * Δt
    end
    particles.v[p] += Δvₚ
    particles.∇v[p] = ∇vₚ
    particles.x[p] += Δxₚ
end
```
