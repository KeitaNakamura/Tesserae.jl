# # [Multi-threading simulation](@id taylor_impact_tutorial)
#
# ```@raw html
# <img src="https://github.com/user-attachments/assets/cd5b926f-4613-4dd5-ac54-181c80b9d8b7" width="400"/>
# ```
#
# | # Particles | # Iterations | Execution time (w/o output) |
# | ----------- | ------------ | ----------------------------|
# | 1.5M        | 1.8k         | 5 min (8 threads)           |
#
# The VTK output is written to `output/taylor_impact`.
#
# This tutorial uses a Taylor impact test to demonstrate multi-threaded MPM simulation.
#

# ## Taylor impact simulation

using Tesserae

using StableRNGs
using TimerOutputs

function main()

    ## Simulation parameters
    t_stop = 80e-6 # Final time
    CFL = 0.8 # Courant number
    if @isdefined(RUN_TESTS) && RUN_TESTS #src
        t_stop = 0.1e-6                   #src
    end                                   #src

    ## Material constants
    E  = 117e9                  # Young's modulus
    ν  = 0.35                   # Poisson's ratio
    λ  = (E*ν) / ((1+ν)*(1-2ν)) # Lame's first parameter
    μ  = E / 2(1 + ν)           # Shear modulus
    ρ⁰ = 8.93e3                 # Initial density
    H  = 0.1e9                  # Hardening parameter
    τ̄y⁰ = 0.4e9                 # Initial yield stress

    ## Geometry parameters for rod
    R = 0.0032
    L = 0.0324

    GridProp = @NamedTuple begin
        x  :: Vec{3, Float64}
        m  :: Float64
        v  :: Vec{3, Float64}
        vⁿ :: Vec{3, Float64}
        mv :: Vec{3, Float64}
        f  :: Vec{3, Float64}
    end
    ParticleProp = @NamedTuple begin
        x  :: Vec{3, Float64}
        m  :: Float64
        V  :: Float64
        v  :: Vec{3, Float64}
        ∇v :: SecondOrderTensor{3, Float64, 9}
        σ  :: SymmetricSecondOrderTensor{3, Float64, 6}
        F  :: SecondOrderTensor{3, Float64, 9}
        c  :: Float64
        ε̄ᵖ :: Float64
        Cᵖ⁻¹ :: SymmetricSecondOrderTensor{3, Float64, 6}
    end

    ## Background grid
    grid = generate_grid(GridProp, CartesianMesh(R/12, (-3R,3R), (-3R,3R), (0,L+0.1L); warn=false))
    if @isdefined(RUN_TESTS) && RUN_TESTS                                                  #src
        grid = generate_grid(GridProp, CartesianMesh(R/6, (-3R,3R), (-3R,3R), (0,L+0.1L); warn=false)) #src
    end                                                                                    #src

    ## Particles
    block = extract(grid.x, (-R,R), (-R,R), (0,L))
    particles = generate_particles(ParticleProp, block; alg=PoissonDiskSampling(spacing=1/3, rng=StableRNG(1234)))
    particles.V .= volume(block) / length(particles)
    filter!(pt -> pt.x[1]^2 + pt.x[2]^2 < R^2, particles)
    @. particles.m = ρ⁰ * particles.V
    @. particles.F = one(particles.F)
    @. particles.Cᵖ⁻¹ = one(particles.Cᵖ⁻¹)
    particles.v .= Ref([0,0,-227]) # Set initial velocity
    @show length(particles)

    ## Basis weights
    weights = generate_basis_weights(KernelCorrection(BSpline(Quadratic())), grid.x, length(particles))

    ## Thread partitioning for multi-threaded P2G transfer
    if Threads.nthreads() == 1
        partition = nothing
    else
        partition = ThreadPartition(grid.x)
    end

    ## Paraview output setup
    outdir = mkpath(joinpath("output", "taylor_impact"))
    pvdfile = joinpath(outdir, "paraview")
    closepvd(openpvd(pvdfile)) # create file

    t = 0.0
    step = 0
    fps = 300e3
    if @isdefined(RUN_TESTS) && RUN_TESTS #src
        fps = inv(t_stop)                 #src
    end                                   #src
    savepoints = collect(LinRange(t, t_stop, round(Int, t_stop*fps)+1))

    reset_timer!()
    Tesserae.@showprogress while t < t_stop

        @timeit "Update timestep" begin
            @threaded for p in eachindex(particles)
                particles.c[p] = sqrt((λ+2μ) / (particles.m[p]/particles.V[p])) + norm(particles.v[p])
            end
            ## `@timeit` can box variables assigned inside it; `::Float64` keeps `Δt` concrete below.
            Δt::Float64 = CFL * spacing(grid.x) / maximum(particles.c)
        end

        @timeit "Update basis weights" begin
            update!(weights, particles, grid.x) # Automatically uses multi-threading
        end

        if partition !== nothing
            @timeit "Update thread partition" begin
                update!(partition, particles.x)
            end
        end

        @timeit "P2G transfer" begin
            @threaded @P2G grid=>i particles=>p weights=>ip partition begin
                m[i]  = @∑ w[ip] * m[p]
                mv[i] = @∑ w[ip] * m[p] * v[p]
                f[i]  = @∑ -V[p] * σ[p] * ∇w[ip]
            end
        end

        @timeit "Grid computation" begin
            @. grid.vⁿ = grid.mv / grid.m * !iszero(grid.m)
            @. grid.v  = grid.vⁿ + Δt * grid.f / grid.m * !iszero(grid.m)
        end

        @timeit "Apply boundary conditions" begin
            for i in eachindex(grid)[:,:,1]
                grid.vⁿ[i] = grid.vⁿ[i] .* (true,true,false)
                grid.v[i] = grid.v[i] .* (true,true,false)
            end
        end

        @timeit "G2P transfer" begin
            @threaded @G2P grid=>i particles=>p weights=>ip begin
                v[p] += @∑ w[ip] * (v[i] - vⁿ[i])
                ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
                x[p] += @∑ w[ip] * v[i] * Δt
            end
        end

        @timeit "Particle computation" begin
            @threaded for p in eachindex(particles)
                ΔFₚ = I + Δt*particles.∇v[p]
                Fₚ = ΔFₚ * particles.F[p]
                σₚ, Cᵖ⁻¹ₚ, ε̄ᵖₚ = vonmises_model(particles.Cᵖ⁻¹[p], particles.ε̄ᵖ[p], Fₚ; λ, μ, H, τ̄y⁰)
                particles.σ[p] = σₚ
                particles.F[p] = Fₚ
                particles.V[p] = det(ΔFₚ) * particles.V[p]
                particles.Cᵖ⁻¹[p] = Cᵖ⁻¹ₚ
                particles.ε̄ᵖ[p] = ε̄ᵖₚ
            end
        end

        t += Δt
        step += 1

        if t > first(savepoints)
            @timeit "Write results" begin
                popfirst!(savepoints)
                openpvd(pvdfile; append=true) do pvd
                    openvtm(string(pvdfile, step)) do vtm
                        openvtk(vtm, particles.x) do vtk
                            vtk["velocity"] = particles.v
                            vtk["plastic strain"] = particles.ε̄ᵖ
                        end
                        openvtk(vtm, grid.x) do vtk
                            vtk["velocity"] = grid.v
                        end
                        pvd[t] = vtm
                    end
                end
            end
            if partition !== nothing
                @timeit "Reorder particles" begin
                    reorder_particles!(particles, partition)
                end
            end
        end
    end
    print_timer()
    sum(particles.x) / length(particles) #src
end

# ## von Mises material model

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
        ν = τ′ᵗʳ / (sqrt(2/3) * norm(τ′ᵗʳ)) # Direction vector
        Δγ = fᵗʳ / (3μ + H)                 # Incremental plastic multiplier
        Δεᵖ = Δγ * dfdσ                     # Incremental logarithmic plastic stretch
        λᵉ = @. exp(log(λᵉᵗʳ) - Δεᵖ)        # Elastic stretch
        τ′ = τ′ᵗʳ - 2μ*Δεᵖ                  # Return map
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

#
# ## Performance
#
# The following results were obtained using 8 threads (started with `julia -t8`).
#
# ### Intel Core Ultra 9 285K
#
# ```julia
# julia> versioninfo()
# Julia Version 1.12.6
# Commit 15346901f00 (2026-04-09 19:20 UTC)
# Build Info:
#   Official https://julialang.org release
# Platform Info:
#   OS: Linux (x86_64-linux-gnu)
#   CPU: 24 × Intel(R) Core(TM) Ultra 9 285K
#   WORD_SIZE: 64
#   LLVM: libLLVM-18.1.7 (ORCJIT, arrowlake-s)
#   GC: Built with stock GC
# Threads: 8 default, 1 interactive, 8 GC (on 24 virtual cores)
#
# julia> main()
# length(particles) = 1472942
# Progress: 100%|█████████████████████████████████████████| Time: 0:08:03
#    Iterations: 1,833
# Wall time/step:
#   Range (min … max): 0.22  s … 8.60  s
#   Mean ± σ:          0.26  s ± 0.32  s
# ──────────────────────────────────────────────────────────────────────────────────────
#                                              Time                    Allocations
#                                     ───────────────────────   ────────────────────────
#          Tot / % measured:                484s /  99.8%           17.5GiB /  99.6%
#
# Section                     ncalls     time    %tot     avg     alloc    %tot      avg
# ──────────────────────────────────────────────────────────────────────────────────────
# Particle computation         1.83k     179s   37.1%  97.8ms    406MiB    2.3%   227KiB
# Update basis weights         1.83k    87.4s   18.1%  47.7ms    681MiB    3.8%   380KiB
# P2G transfer                 1.83k    72.5s   15.0%  39.5ms    152MiB    0.8%  84.7KiB
# G2P transfer                 1.83k    62.0s   12.8%  33.8ms   46.5MiB    0.3%  26.0KiB
# Write results                   25    55.1s   11.4%   2.20s   8.16GiB   46.8%   334MiB
# Grid computation             1.83k    10.0s    2.1%  5.48ms   79.9MiB    0.4%  44.6KiB
# Update timestep              1.83k    5.77s    1.2%  3.15ms   61.7MiB    0.3%  34.5KiB
# Apply boundary conditions    1.83k    4.93s    1.0%  2.69ms   7.49GiB   42.9%  4.19MiB
# Update thread partition      1.83k    4.24s    0.9%  2.31ms   79.1MiB    0.4%  44.2KiB
# Reorder particles               25    1.64s    0.3%  65.7ms    329MiB    1.8%  13.1MiB
# ──────────────────────────────────────────────────────────────────────────────────────
# ```
#
# ### Apple M2 Ultra
#
# ```julia
# julia> versioninfo()
# Julia Version 1.12.6
# Commit 15346901f00 (2026-04-09 19:20 UTC)
# Build Info:
#   Official https://julialang.org release
# Platform Info:
#   OS: macOS (arm64-apple-darwin24.0.0)
#   CPU: 24 × Apple M2 Ultra
#   WORD_SIZE: 64
#   LLVM: libLLVM-18.1.7 (ORCJIT, apple-m2)
#   GC: Built with stock GC
# Threads: 8 default, 1 interactive, 8 GC (on 16 virtual cores)
#
# julia> main()
# length(particles) = 1472942
# Progress: 100%|█████████████████████████████████████████| Time: 0:06:11
#    Iterations: 1,833
# Wall time/step:
#   Range (min … max): 0.15  s … 8.95  s
#   Mean ± σ:          0.20  s ± 0.42  s
# ──────────────────────────────────────────────────────────────────────────────────────
#                                              Time                    Allocations
#                                     ───────────────────────   ────────────────────────
#          Tot / % measured:                371s /  99.8%           17.6GiB /  99.4%
#
# Section                     ncalls     time    %tot     avg     alloc    %tot      avg
# ──────────────────────────────────────────────────────────────────────────────────────
# Particle computation         1.83k     198s   53.5%   108ms    408MiB    2.3%   228KiB
# Write results                   25    81.3s   22.0%   3.25s   8.19GiB   46.8%   335MiB
# P2G transfer                 1.83k    27.4s    7.4%  14.9ms    153MiB    0.9%  85.4KiB
# G2P transfer                 1.83k    27.1s    7.3%  14.8ms   48.0MiB    0.3%  26.8KiB
# Update basis weights         1.83k    17.7s    4.8%  9.66ms    695MiB    3.9%   388KiB
# Grid computation             1.83k    9.58s    2.6%  5.23ms   81.2MiB    0.5%  45.4KiB
# Apply boundary conditions    1.83k    5.16s    1.4%  2.81ms   7.50GiB   42.8%  4.19MiB
# Update thread partition      1.83k    1.76s    0.5%   961μs   79.6MiB    0.4%  44.5KiB
# Update timestep              1.83k    1.57s    0.4%   858μs   61.1MiB    0.3%  34.1KiB
# Reorder particles               25    851ms    0.2%  34.0ms    341MiB    1.9%  13.6MiB
# ──────────────────────────────────────────────────────────────────────────────────────
# ```
#
# ## Scalability
#
# The scalability depends strongly on the memory system. Apple's
# [M2 Ultra](https://www.apple.com/newsroom/2023/06/apple-introduces-m2-ultra/)
# provides 800 GB/s of unified memory bandwidth, whereas Intel's
# [Core Ultra 9 285K](https://www.intel.com/content/www/us/en/products/sku/241060/intel-core-ultra-9-processor-285k-36m-cache-up-to-5-70-ghz/specifications.html)
# specifies two DDR5-6400 memory channels, corresponding to about 102 GB/s of
# theoretical peak bandwidth. This difference matters here because the basis-weight
# update and P2G/G2P transfer kernels perform many reads and writes compared with the
# amount of arithmetic work, so they can become limited by memory bandwidth. The
# particle constitutive update is more compute-heavy and tends to scale better with
# thread count.
#
# ### Intel Core Ultra 9 285K
#
# ```@example
# using Plots                                             # hide
# plot(xlabel = "Number of threads", ylabel = "Speedup",  # hide
#      xlims = (0,18), ylims = (0,18), palette = :RdBu_4) # hide
# plot!([1, 2, 4, 8, 16],                                 # hide
#       179.0 ./ [179.0, 108.0, 98.3, 87.4, 85.5],        # hide
#       label = "Update basis weights", marker = "o")     # hide
# plot!([1, 2, 4, 8, 16],                                 # hide
#       224.0 ./ [224.0, 152.0, 93.4, 72.5, 63.0],        # hide
#       label = "P2G transfer", marker = "o")             # hide
# plot!([1, 2, 4, 8, 16],                                 # hide
#       195.0 ./ [195.0, 126.0, 76.2, 62.0, 54.7],        # hide
#       label = "G2P transfer", marker = "o")             # hide
# plot!([1, 2, 4, 8, 16],                                 # hide
#       1350.0 ./ [1350.0, 678.0, 349.0, 179.0, 131.0],   # hide
#       label = "Particle computation", marker = "o")     # hide
# plot!([1, 2, 4, 8, 16],                                 # hide
#       1971.5 ./ [1971.5, 1093.3, 641.2, 428.9, 362.3],  # hide
#       label = "Total (w/o output)", color = "black",    # hide
#       marker = "o")                                     # hide
# plot!([1, 17], [1, 17],                                 # hide
#       color = "black", linestyle = :dash,               # hide
#       label = "")                                       # hide
# ```
#
# ### Apple M2 Ultra
#
# ```@example
# using Plots                                             # hide
# plot(xlabel = "Number of threads", ylabel = "Speedup",  # hide
#      xlims = (0,18), ylims = (0,18), palette = :RdBu_4) # hide
# plot!([1, 2, 4, 8, 16],                                 # hide
#       101.0 ./ [101.0, 56.0, 32.7, 17.7, 13.1],         # hide
#       label = "Update basis weights", marker = "o")     # hide
# plot!([1, 2, 4, 8, 16],                                 # hide
#       146.0 ./ [146.0, 83.2, 45.8, 27.4, 19.9],         # hide
#       label = "P2G transfer", marker = "o")             # hide
# plot!([1, 2, 4, 8, 16],                                 # hide
#       197.0 ./ [197.0, 102.0, 51.8, 27.1, 15.5],        # hide
#       label = "G2P transfer", marker = "o")             # hide
# plot!([1, 2, 4, 8, 16],                                 # hide
#       1450.0 ./ [1450.0, 737.0, 391.0, 198.0, 106.0],   # hide
#       label = "Particle computation", marker = "o")     # hide
# plot!([1, 2, 4, 8, 16],                                 # hide
#       1915.8 ./ [1915.8, 1003.0, 543.2, 289.7, 174.0],  # hide
#       label = "Total (w/o output)", color = "black",    # hide
#       marker = "o")                                     # hide
# plot!([1, 17], [1, 17],                                 # hide
#       color = "black", linestyle = :dash,               # hide
#       label = "")                                       # hide
# ```

using Test                                 #src
if @isdefined(RUN_TESTS) && RUN_TESTS      #src
    @test main() ≈ [0,0,0.01622] rtol=1e-3 #src
end                                        #src
