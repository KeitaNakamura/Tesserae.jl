# GPU computing

GPU support in Tesserae is built around the same transfer notation used on CPU.
Once the mesh, grid, particles, and basis weights are moved to a GPU backend, macros such as [`@P2G`](@ref) and [`@G2P`](@ref) launch GPU kernels instead of CPU loops.
This keeps the MPM update close to the CPU version while moving the particle-grid work to the GPU.

The code still needs to follow GPU array rules.
Grid and particle updates should be written as transfer macros, broadcasts, or GPU kernels, and data should be copied back to CPU memory only for inspection or output.

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
Scalar indexing from CPU code falls back to the CPU and is disallowed in non-interactive CUDA.jl execution; see CUDA.jl's [scalar indexing workflow](https://cuda.juliagpu.org/stable/usage/workflow/#UsageWorkflowScalar) for details:

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
for i in eachindex(grid)[:,:,begin]
    grid.v[i] = grid.v[i] .* (true, true, false)
end
```

On GPU, write the same operation with [`@foreach`](@ref), which dispatches the
boundary slice as a GPU kernel:

```julia
@foreach grid_gpu[:,:,begin]=>i begin
    v[i] = v[i] .* (true, true, false)
end
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

By default GPU `@P2G` uses particle-parallel kernels with atomic updates.
Passing a [`Partition`](@ref) that has been moved to the device with `gpu` selects a block-scheduled path instead, which accumulates each grid block into shared memory before touching the grid.
Only `@P2G` has that path; `@G2P` and `@G2P2G` take no partition on GPU.

CPU threaded scattering:

```julia
partition = Partition(grid.x)
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

## Assembled matrices on GPU

`@P2G_Matrix` assembles into a sparse matrix that lives on the device.
Build the matrix on the CPU with [`create_sparse_matrix`](@ref) and move it with the same call as everything else:

```julia
A = create_sparse_matrix(T, basis, mesh; ndofs=2)
grid, particles, weights, A = (grid, particles, weights, A) .|> gpu_preserve

@P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
    A[i,j] = @∑ ∇w[ip] ⊡ c[p] ⊡ ∇w[jp] * V[p]
end
```

The transfer is particle-parallel and accumulates with atomics, so the summation order within a stored entry is not reproducible between runs, exactly as for GPU `@P2G`.
A [`Partition`](@ref) selects a block-scheduled path for `@P2G`, but there is no such path for `@P2G_Matrix`; pass no partition.

The target must be a plain sparse matrix over a Cartesian mesh, and it must come from `create_sparse_matrix`.
On the CPU the canonical sparsity pattern is checked from the stored entry count of every column; on the device only the total is checked, because walking the stored rows would cost a kernel launch and a readback on every call.
Views taken with `view` and the block views from [`create_block_sparse_matrix`](@ref) are CPU-only.
`@P2G_Matrix` reads the stored basis weights, so weights generated with `deferred=true` are rejected on GPU as they are on the CPU.

Two things are still missing before an assembled GPU system can be solved end to end:

- [`extract`](@ref) reduces the matrix to the active DoFs by indexing it with a vector of DoF numbers, which CUDA's sparse matrices do not support; it raises a scalar-indexing error on a device matrix.
- `\` is not defined for CUDA's sparse matrices, so `Tesserae.newton!` needs an explicit `linsolve` on GPU, using a solver package such as CUDSS.jl or Krylov.jl.

## Taylor impact on GPU

This section rewrites the [Taylor impact tutorial](@ref taylor_impact_tutorial) as a GPU simulation.
The transfer equations and the von Mises material model are unchanged; only the execution pattern is adjusted.
The main changes are:

- Remove `@threaded`, which has no GPU form. A [`Partition`](@ref) may be kept, but must be moved with `gpu` and updated after the move; `reorder_particles!` works on the moved partition too, though particles outside the mesh are an error there instead of being kept at the end of the array.
- Move the simulation objects to GPU with `gpu_preserve` after CPU-side setup.
- Generate the basis weights with `deferred=true` (see [Deferred basis weights](@ref)): evaluating the values inside the transfers outruns reading stored ones on GPU, and the weights then occupy no memory.
- Keep grid and particle calculations inside GPU operations, using `@P2G`, `@G2P`, and `@foreach`.
- Rewrite the slip floor boundary condition with a boundary-slice `@foreach` loop to avoid scalar indexing on GPU arrays.
- Copy data back with `cpu` only when writing VTK output.

For reference, the compute-only runtime on an NVIDIA GeForce RTX 5090, excluding VTK output and the first-call kernel compilation, is:

| Precision | # Particles | # Iterations | Execution time (w/o output) |
| --------- | ----------- | ------------ | ---------------------------- |
| Float64   | 1.48M       | 1.8k         | 31 sec                       |
| Float32   | 1.48M       | 1.8k         | 8 sec                        |

The VTK output is written to `output/taylor_impact_gpu`.

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

    ## Basis weights, evaluated inside the transfers instead of being stored
    weights = generate_basis_weights(T, KernelCorrection(BSpline(Quadratic())), grid.x, length(particles); deferred=true)

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

            @foreach grid[:,:,begin]=>i begin
                vⁿ[i] = vⁿ[i] .* (true, true, false)
                v[i] = v[i] .* (true, true, false)
            end

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

## Implicit MPM on GPU

This section rewrites the [Jacobian-free Newton--Krylov tutorial](@ref implicit_jacobian_free_tutorial) as a GPU simulation.
The residual and the Jacobian-vector product are unchanged; what changes is how the free degrees of freedom are selected.
The main changes are:

- Carry the DoF mask as a `Vec{ndofs, Bool}` grid field and write it with `@foreach`, instead of building a host `BitArray` in a scalar loop. [`DofMap`](@ref) reads such a field directly, so `free(grid.u)` returns a device view and the whole Newton loop stays on GPU. A device Boolean array of size `(ndofs, size(grid)...)` works just as well when the mask should not live on the grid.
- Rewrite the boundary conditions as boundary-slice `@foreach` loops to avoid scalar indexing on GPU arrays.
- Move the simulation objects with `gpu_preserve`. A Jacobian-free Krylov solve converges on the residual norm, and `Float32` limits how far that can be driven.
- Give `LinearOperator` the device vector type through its `S` keyword, so that `Krylov.gmres` allocates its workspace on the device.
- Copy data back with `cpu` only when writing VTK output.

For reference, the runtime on an NVIDIA GeForce RTX 5090, excluding VTK output and the first-call kernel compilation, is:

| Precision | # Particles | # Iterations | Execution time (w/o output) |
| --------- | ----------- | ------------ | ---------------------------- |
| Float64   | 26k         | 300          | 12 sec                       |

The VTK output is written to `output/implicit_jacobian_free_gpu`.

```julia
using Tesserae
using CUDA

using Krylov: gmres
using LinearOperators: LinearOperator

function main()
    T = Float64

    ## Simulation parameters
    h  = T(0.05)   # Grid spacing
    t_stop = T(3)  # Final time
    Δt = T(0.01)   # Time step

    ## Material constants
    E  = T(100e3)               # Young's modulus
    ν  = T(0.3)                 # Poisson's ratio
    λ  = (E*ν) / ((1+ν)*(1-2ν)) # Lame's first parameter
    μ  = E / 2(1 + ν)           # Shear modulus
    ρ⁰ = T(1000)                # Initial density

    ## Newmark-beta integration
    β = T(1/4)
    γ = T(1/2)

    GridProp = @NamedTuple begin
        X    :: Vec{3, T}
        m    :: T
        m⁻¹  :: T
        v    :: Vec{3, T}
        vⁿ   :: Vec{3, T}
        mv   :: Vec{3, T}
        a    :: Vec{3, T}
        aⁿ   :: Vec{3, T}
        ma   :: Vec{3, T}
        u    :: Vec{3, T}
        f    :: Vec{3, T}
        δu   :: Vec{3, T}
        free :: Vec{3, Bool}
    end
    ParticleProp = @NamedTuple begin
        x    :: Vec{3, T}
        m    :: T
        V⁰   :: T
        v    :: Vec{3, T}
        a    :: Vec{3, T}
        ∇u   :: SecondOrderTensor{3, T, 9}
        F    :: SecondOrderTensor{3, T, 9}
        ΔF⁻¹ :: SecondOrderTensor{3, T, 9}
        τ    :: SecondOrderTensor{3, T, 9}
        ℂ    :: FourthOrderTensor{3, T, 81}
    end

    ## Background grid
    grid = generate_grid(GridProp, CartesianMesh(T, h, (0,1.5), (-0.6,0.6), (-0.6,0.6)))

    ## Particles
    beam = extract(grid.X, (0,1.5), (-0.3,0.3), (-0.3,0.3))
    particles = generate_particles(ParticleProp, beam; alg=GridSampling(spacing=1/6))
    particles.V⁰ .= volume(beam) / length(particles)
    filter!(particles) do pt
        x, y, z = pt.x
        (-0.3<y<-0.25 || 0.25<y<0.3) && (-0.3<z<-0.25 || 0.25<z<0.3)
    end
    @. particles.m = ρ⁰ * particles.V⁰
    @. particles.F = one(particles.F)
    @show length(particles)

    ## Basis weights
    weights = generate_basis_weights(T, KernelCorrection(BSpline(Quadratic())), grid.X, length(particles))

    ## Neo-Hookean model
    function kirchhoff_stress(F)
        J = det(F)
        b = symmetric(F * F')
        μ*(b-I) + λ*log(J)*I
    end

    ## Paraview output setup
    outdir = mkpath(joinpath("output", "implicit_jacobian_free_gpu"))
    pvdfile = joinpath(outdir, "paraview")
    closepvd(openpvd(pvdfile)) # create file

    t = zero(T)
    step = 0
    fps = 60
    savepoints = collect(LinRange(t, t_stop, round(Int, t_stop*fps)+1))

    ## Move the simulation state to the GPU after CPU-side setup; the time loop below stays on GPU.
    let (grid, particles, weights) = (grid, particles, weights) .|> gpu_preserve

        Tesserae.@showprogress while t < t_stop

            update!(weights, particles, grid.X)

            @P2G grid=>i particles=>p weights=>ip begin
                m[i]  = @∑ w[ip] * m[p]
                mv[i] = @∑ w[ip] * m[p] * v[p]
                ma[i] = @∑ w[ip] * m[p] * a[p]
            end

            ## Compute the grid velocity and acceleration at t = tⁿ
            @. grid.m⁻¹ = inv(grid.m) * !iszero(grid.m)
            @. grid.vⁿ = grid.mv * grid.m⁻¹
            @. grid.aⁿ = grid.ma * grid.m⁻¹

            ## Mark the free DoFs and update the boundary conditions on the device
            @foreach grid=>i begin
                u[i] = zero(u[i])
                movable = !iszero(m[i])
                free[i] = Vec(movable, movable, movable)
            end
            @foreach grid[begin,:,:]=>i begin
                free[i] = zero(free[i])
            end
            @foreach grid[end,:,:]=>i begin
                free[i] = zero(free[i])
                u[i] = $(rotmat(2π*Δt, Vec(T(1),T(0),T(0))) - I) * X[i]
            end
            free = DofMap(grid.free)

            ## Solve the nonlinear equation
            state = (; grid, particles, weights, kirchhoff_stress, β, γ, free, Δt)
            U = copy(free(grid.u)) # Convert grid data to plain vector data
            compute_residual(U) = residual(U, state)
            compute_jacobian(U) = jacobian(U, state)
            Tesserae.newton!(U, compute_residual, compute_jacobian;
                             linsolve = (x,A,b)->copy!(x,gmres(A,b)[1]))

            @G2P grid=>i particles=>p weights=>ip begin
                v[p] += @∑ w[ip] * ((1-γ)*a[p] + γ*a[i]) * Δt
                a[p]  = @∑ w[ip] * a[i]
                x[p]  = @∑ w[ip] * (X[i] + u[i])
                ∇u[p] = @∑ u[i] ⊗ ∇w[ip]
                F[p]  = (I + ∇u[p]) * F[p]
            end

            t += Δt
            step += 1

            if t > first(savepoints)
                popfirst!(savepoints)
                openpvd(pvdfile; append=true) do pvd
                    openvtk(string(pvdfile, step), cpu(particles.x)) do vtk
                        τ, F = cpu(particles.τ), cpu(particles.F)
                        vtk["Velocity (m/s)"] = cpu(particles.v)
                        vtk["von Mises stress (kPa)"] = @. 1e-3 * vonmises(τ / det(F))
                        pvd[t] = vtk
                    end
                end
            end
        end
    end
end

function residual(U::AbstractVector, state)
    (; grid, particles, weights, kirchhoff_stress, β, γ, free, Δt) = state

    free(grid.u) .= U
    @. grid.a = (1/(β*Δt^2))*grid.u - (1/(β*Δt))*grid.vⁿ - (1/2β-1)*grid.aⁿ
    @. grid.v = grid.vⁿ + ((1-γ)*grid.aⁿ + γ*grid.a) * Δt

    geometric(τ) = @einsum (i,j,k,l) -> τ[i,l] * one(τ)[j,k]
    @G2P2G grid=>i particles=>p weights=>ip begin
        ∇u[p] = @∑ u[i] ⊗ ∇w[ip]
        ΔF⁻¹[p] = inv(I + ∇u[p])
        F = (I + ∇u[p]) * F[p]
        ∂τ∂F, τ = gradient(kirchhoff_stress, F, :all)
        τ[p] = τ
        ℂ[p] = ∂τ∂F ⊡ F' - geometric(τ)
        f[i] = @∑ V⁰[p] * τ[p] * (∇w[ip] ⊡ ΔF⁻¹[p])
    end

    @. β*Δt^2 * ($free(grid.a) + $free(grid.f) * $free(grid.m⁻¹))
end

function jacobian(U::AbstractVector, state)
    (; grid, particles, weights, β, free, Δt) = state

    fillzero!(grid.δu)
    function mul!(JδU, δU)
        free(grid.δu) .= δU

        @G2P2G grid=>i particles=>p weights=>ip begin
            ∇u[p] = @∑ δu[i] ⊗ (∇w[ip] ⊡ ΔF⁻¹[p])
            τ[p] = ℂ[p] ⊡₂ ∇u[p]
            f[i] = @∑ V⁰[p] * τ[p] * (∇w[ip] ⊡ ΔF⁻¹[p])
        end

        @. JδU = δU + β*Δt^2 * $free(grid.f) * $free(grid.m⁻¹)
    end

    U = free(grid.u)
    LinearOperator(eltype(U), ndofs(free), ndofs(free), false, false, mul!; S = typeof(similar(U, 0)))
end
```
