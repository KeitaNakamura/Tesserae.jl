# GPU computing

Tesserae can move grids, particles, basis weights, and meshes to a GPU with `gpu`.
The transfer macros then dispatch to GPU kernels when all inputs live on the same GPU backend.

Load the GPU backend package together with Tesserae:

```julia
using Tesserae
using CUDA # or Metal
```

## Basic workflow

Choose the floating-point type first, then build the mesh, grid, particles, and weights from that type.
This keeps the code close to the CPU version while avoiding mixed precision inside GPU kernels.

```julia
using Tesserae
using CUDA

const T = Float32

GridProp = @NamedTuple begin
    x  :: Vec{2,T}
    m  :: T
    m⁻¹:: T
    mv :: Vec{2,T}
    v  :: Vec{2,T}
end

ParticleProp = @NamedTuple begin
    x :: Vec{2,T}
    m :: T
    v :: Vec{2,T}
end

mesh = CartesianMesh(T, T(0.02), (-1, 1), (0, 3))
grid = generate_grid(GridProp, mesh)
particles = generate_particles(ParticleProp, grid.x)
@. particles.m = 1
@. particles.v = zero(particles.v)
weights = generate_basis_weights(T, BSpline(Quadratic()), grid.x, length(particles))

grid_gpu = gpu(grid);
particles_gpu = gpu(particles);
weights_gpu = gpu(weights);
```

After that, the transfer code is the same as the CPU version.
A single update step can be written as:

```julia
function step!(grid_gpu, particles_gpu, weights_gpu, dt)
    update!(weights_gpu, particles_gpu, grid_gpu.x)

    @P2G grid_gpu=>i particles_gpu=>p weights_gpu=>ip begin
        m[i]  = @∑ w[ip] * m[p]
        mv[i] = @∑ w[ip] * m[p] * v[p]
        m⁻¹[i] = ifelse(iszero(m[i]), zero(m[i]), inv(m[i]))
        v[i] = mv[i] * m⁻¹[i]
    end

    @G2P grid_gpu=>i particles_gpu=>p weights_gpu=>ip begin
        v[p] = @∑ v[i] * w[ip]
        x[p] += v[p] * dt
    end
end

step!(grid_gpu, particles_gpu, weights_gpu, T(1.0e-4))
```

Use `cpu` to copy GPU data back to CPU memory, for example for output:

```julia
x = cpu(particles_gpu.x)
```

## GPU arrays

After calling `gpu`, grid fields, particle fields, basis weights, and mesh coordinates in the returned objects are GPU arrays.
Scalar indexing from CPU code is not allowed:

```julia
julia> x = gpu(rand(3))
3-element CuArray{Float32, 1, CUDA.DeviceMemory}:
 0.7683449
 0.4430822
 0.88063353

julia> x[1]
ERROR: Scalar indexing is disallowed.
```

The same rule applies to fields such as `grid_gpu.v`, `particles_gpu.x`, and `weights_gpu`.

!!! note
    In the REPL, displaying an object that contains GPU arrays, such as `particles_gpu` or `grid_gpu`, can hit the same scalar indexing error.
    Suppress display with `;` when assigning GPU objects:

    ```julia
    particles_gpu = gpu(particles);
    ```

    Copy data back with `cpu` before inspecting values on the CPU.

Use array operations such as broadcasting or `map!`, or write the operation inside a transfer macro or a GPU kernel.
Indexing inside `@P2G` and `@G2P` is fine because the generated code runs in GPU kernels.
This is also how boundary conditions should be applied on GPU.
For example, on a 3D grid, a CPU slip floor boundary condition can be written as

```julia
for i in eachindex(grid)[:, :, begin]
    grid.v[i] = grid.v[i] .* (true, true, false)
end
```

On GPU, write the same operation as a broadcast:

```julia
slip_floor(v) = v .* (true, true, false)

@. grid_gpu.v[:, :, begin] = slip_floor(grid_gpu.v[:, :, begin])
```

## Floating-point type

`gpu` returns GPU arrays and, by default, converts floating-point arrays to `Float32`.
Use `gpu_preserve` instead when the original floating-point type should be preserved on the GPU.
The conversion applies to arrays and adapted Tesserae objects.
It does not rewrite scalar constants captured by a kernel, so write scalar constants with the intended type:

```julia
T = Float32

dt = T(1.0e-4)
gravity = Vec(T(0), T(-9.81))
```

This matters on Metal, where `Float64` cannot be used in GPU kernels.

## Transfers

GPU `@P2G` uses particle-parallel kernels with atomic updates.
Do not pass a [`ThreadPartition`](@ref) to GPU transfers; `ThreadPartition` is a CPU scheduling tool for threaded scattering.

CPU threaded scattering:

```julia
partition = ThreadPartition(grid.x)
update!(partition, particles.x)

@threaded @P2G grid=>i particles=>p weights=>ip partition begin
    m[i] = @∑ w[ip] * m[p]
end
```

GPU scattering:

```julia
@P2G grid_gpu=>i particles_gpu=>p weights_gpu=>ip begin
    m[i] = @∑ w[ip] * m[p]
end
```

`@G2P` is also dispatched to GPU kernels when the grid, particles, and weights are GPU objects.

## SpArray on GPU

[`SpArray`](@ref) can also be used on GPU.
Create the sparse grid on CPU, move it to the GPU, and update its sparsity directly from particle positions.
The same scalar-indexing rule applies: use `SpArray` through `update_sparsity!`, transfer macros, broadcasts, or GPU kernels rather than CPU loops over individual entries.

```julia
grid = generate_grid(SpArray, GridProp, mesh)
weights = generate_basis_weights(T, BSpline(Quadratic()), grid.x, length(particles))

grid_gpu = gpu(grid);
particles_gpu = gpu(particles);
weights_gpu = gpu(weights);

update_sparsity!(grid_gpu, particles_gpu.x)
update!(weights_gpu, particles_gpu, grid_gpu.x)

@P2G grid_gpu=>i particles_gpu=>p weights_gpu=>ip begin
    m[i] = @∑ w[ip] * m[p]
end
```

Call `update_sparsity!(grid_gpu, particles_gpu.x)` again after particle positions have moved and before the next transfer.
This keeps the active blocks large enough for the particle support nodes.

On GPU, `SpArray` mainly reduces grid-field storage and grid-wide operations over inactive regions.
It should not be expected to remove the main cost of `@P2G`, which is still proportional to the number of particles times the number of support nodes.

## Taylor impact on GPU

This section rewrites the [Taylor impact tutorial](@ref taylor_impact_tutorial) as a GPU simulation.
The transfer equations and the von Mises material model are unchanged; only the execution pattern is adjusted.
The main changes are:

- Remove CPU threading utilities such as `@threaded`, `ThreadPartition`, and `reorder_particles!`.
- Move the simulation objects to GPU with `gpu_preserve` after CPU-side setup.
- Keep grid and particle calculations inside GPU operations, using `@P2G`, `@G2P`, and broadcasts.
- Rewrite the slip floor boundary condition as a broadcast to avoid scalar indexing on GPU arrays.
- Copy data back with `cpu` only when writing VTK output.

The following compute-only results were obtained on an NVIDIA GeForce RTX 5090 by disabling VTK output.

| Precision | # Particles | # Iterations | Execution time (w/o output) |
| --------- | ----------- | ------------ | ---------------------------- |
| Float64   | 1.48M       | 1.8k         | 1 min 01 sec                 |
| Float32   | 1.48M       | 1.8k         | 37 sec                       |

```julia
using Tesserae
using CUDA

function main()
    T = Float64

    ## Simulation parameters
    t_stop = T(80e-6) # Final time
    CFL = T(0.8)     # Courant number

    ## Material constants
    E  = T(117e9)               # Young's modulus
    ν  = T(0.35)                # Poisson's ratio
    λ  = (E*ν) / ((1+ν)*(1-2ν)) # Lame's first parameter
    μ  = E / 2(1 + ν)           # Shear modulus
    ρ⁰ = T(8.93e3)              # Initial density
    H  = T(0.1e9)               # Hardening parameter
    τ̄y⁰ = T(0.4e9)              # Initial yield stress

    ## Geometry parameters for rod
    R = T(0.0032)
    L = T(0.0324)

    GridProp = @NamedTuple begin
        x  :: Vec{3, T}
        m  :: T
        v  :: Vec{3, T}
        vⁿ :: Vec{3, T}
        mv :: Vec{3, T}
        f  :: Vec{3, T}
    end
    ParticleProp = @NamedTuple begin
        x  :: Vec{3, T}
        m  :: T
        V  :: T
        v  :: Vec{3, T}
        ∇v :: SecondOrderTensor{3, T, 9}
        σ  :: SymmetricSecondOrderTensor{3, T, 6}
        F  :: SecondOrderTensor{3, T, 9}
        c  :: T
        ε̄ᵖ :: T
        Cᵖ⁻¹ :: SymmetricSecondOrderTensor{3, T, 6}
    end

    ## Background grid
    grid = generate_grid(GridProp, CartesianMesh(T, R/12, (-3R,3R), (-3R,3R), (0,L+0.1L)))

    ## Particles
    block = extract(grid.x, (-R,R), (-R,R), (0,L))
    particles = generate_particles(ParticleProp, block; alg=PoissonDiskSampling(spacing=1/3))
    particles.V .= volume(block) / length(particles)
    filter!(pt -> pt.x[1]^2 + pt.x[2]^2 < R^2, particles)
    @. particles.m = ρ⁰ * particles.V
    @. particles.F = one(particles.F)
    @. particles.Cᵖ⁻¹ = one(particles.Cᵖ⁻¹)
    particles.v .= Ref(Vec(T(0), T(0), T(-227))) # Set initial velocity

    ## Basis weights
    weights = generate_basis_weights(T, KernelCorrection(BSpline(Quadratic())), grid.x, length(particles))

    ## Paraview output setup
    outdir = mkpath(joinpath("output", "taylor_impact_gpu"))
    pvdfile = joinpath(outdir, "paraview")
    closepvd(openpvd(pvdfile)) # create file

    t = zero(T)
    step = 0
    fps = T(300e3)
    savepoints = collect(LinRange(t, t_stop, round(Int, t_stop*fps)+1))

    # Move the simulation state to the GPU after CPU-side setup; the time loop below stays on GPU.
    let (grid, particles, weights) = (grid, particles, weights) .|> gpu_preserve

        Tesserae.@showprogress while t < t_stop

            @. particles.c = sqrt((λ+2μ) / (particles.m/particles.V)) + norm(particles.v)
            Δt = CFL * spacing(grid.x) / maximum(particles.c)

            update!(weights, particles, grid.x)

            @P2G grid=>i particles=>p weights=>ip begin
                m[i]  = @∑ w[ip] * m[p]
                mv[i] = @∑ w[ip] * m[p] * v[p]
                f[i]  = @∑ -V[p] * σ[p] * ∇w[ip]
                vⁿ[i] = mv[i] / m[i] * !iszero(m[i])
                v[i]  = vⁿ[i] + Δt * f[i] / m[i] * !iszero(m[i])
            end

            slip_floor(v) = v .* (true, true, false)
            @. grid.vⁿ[:, :, begin] = slip_floor(grid.vⁿ[:, :, begin])
            @. grid.v[:, :, begin] = slip_floor(grid.v[:, :, begin])

            @G2P grid=>i particles=>p weights=>ip begin
                v[p] += @∑ w[ip] * (v[i] - vⁿ[i])
                ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
                x[p] += @∑ w[ip] * v[i] * Δt
                ΔFₚ = I + Δt * ∇v[p]
                Fₚ = ΔFₚ * F[p]
                σₚ, Cᵖ⁻¹ₚ, ε̄ᵖₚ = vonmises_model(Cᵖ⁻¹[p], ε̄ᵖ[p], Fₚ; λ, μ, H, τ̄y⁰)
                σ[p] = σₚ
                F[p] = Fₚ
                V[p] = det(ΔFₚ) * V[p]
                Cᵖ⁻¹[p] = Cᵖ⁻¹ₚ
                ε̄ᵖ[p] = ε̄ᵖₚ
            end

            t += Δt
            step += 1

            if t > first(savepoints)
                popfirst!(savepoints)
                openpvd(pvdfile; append=true) do pvd
                    openvtm(string(pvdfile, step)) do vtm
                        openvtk(vtm, cpu(particles.x)) do vtk
                            vtk["velocity"] = cpu(particles.v)
                            vtk["plastic strain"] = cpu(particles.ε̄ᵖ)
                        end
                        openvtk(vtm, cpu(grid.x)) do vtk
                            vtk["velocity"] = cpu(grid.v)
                        end
                        pvd[t] = vtm
                    end
                end
            end
        end
    end
end

function vonmises_model(Cᵖⁿ⁻¹, ε̄ᵖⁿ, F; λ, μ, H, τ̄y⁰)
    κ = λ + 2μ/3                             # Bulk modulus
    J = det(F)                               # Jacobian
    p = κ * log(J) / J                       # Pressure
    bᵉᵗʳ = symmetric(F * Cᵖⁿ⁻¹ * F')         # Trial left Cauchy-Green tensor
    vals, vecs = eigen(bᵉᵗʳ)                 # Eigenvalue decomposition
    λᵉᵗʳ = sqrt.(vals)                       # Trial stretches
    nᵗʳₐ = (vecs[:,1], vecs[:,2], vecs[:,3]) # Principal directions
    τ′ᵗʳ = @. 2μ*log(λᵉᵗʳ) - 2μ/3*log(J)     # Trial Kirchhoff stress

    f(τ) = sqrt(3τ⋅τ/2) - (τ̄y⁰ + H*ε̄ᵖⁿ) # Yield function
    dfdσ, fᵗʳ = gradient(f, τ′ᵗʳ, :all)
    if fᵗʳ > 0
        Δγ = fᵗʳ / (3μ + H)          # Incremental plastic multiplier
        Δεᵖ = Δγ * dfdσ              # Incremental logarithmic plastic stretch
        λᵉ = @. exp(log(λᵉᵗʳ) - Δεᵖ) # Elastic stretch
        τ′ = τ′ᵗʳ - 2μ*Δεᵖ           # Return map
    else # Elastic response
        Δγ = zero(H)
        λᵉ = λᵉᵗʳ
        τ′ = τ′ᵗʳ
    end

    ## Update inverse of elastic left Cauchy-Green tensor
    nₐ = nᵗʳₐ
    bᵉ = mapreduce((λᵉ,nₐ) -> λᵉ^2 * nₐ^⊗(2), +, λᵉ, nₐ)

    ## Update stress
    σ′ = τ′ / J    # Principal deviatoric Cauchy stress
    σ  = @. σ′ + p # Principal Cauchy stress
    σ  = mapreduce((σ,nₐ) -> σ * nₐ^⊗(2), +, σ, nₐ)

    ## Update state variables
    F⁻¹ = inv(F)
    Cᵖ⁻¹ = symmetric(F⁻¹ * bᵉ * F⁻¹') # Update plastic right Cauchy-Green tensor
    ε̄ᵖ = ε̄ᵖⁿ + Δγ                     # Update equivalent plastic strain

    σ, Cᵖ⁻¹, ε̄ᵖ
end
```
