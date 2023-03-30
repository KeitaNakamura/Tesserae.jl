# # Contacting grains
#
# ![](https://user-images.githubusercontent.com/16015926/226637268-78241ae9-1d5d-486f-b3b1-db343dfa8b69.gif)

using Marble
using StableRNGs #src

include(joinpath(pkgdir(Marble), "docs/literate/models/LinearElastic.jl"))

struct GridState
    x  :: Vec{2, Float64}
    m  :: Float64
    mv :: Vec{2, Float64}
    f  :: Vec{2, Float64}
    v  :: Vec{2, Float64}
    vⁿ :: Vec{2, Float64}
    ∇m :: Vec{2, Float64}
end

struct ParticleState
    x  :: Vec{2, Float64}
    m  :: Float64
    V  :: Float64
    v  :: Vec{2, Float64}
    ∇v :: SecondOrderTensor{3, Float64, 9}
    σ  :: SymmetricSecondOrderTensor{3, Float64, 6}
    l  :: Float64
end

function contacting_grains(
        itp::Interpolation = QuadraticBSpline(),
        alg::TransferAlgorithm = FLIP(),
        ;output::Bool = true, #src
        test::Bool = false,   #src
    )

    ## simulation parameters
    CFL    = 0.5   # Courant number
    t_stop = 1.2   # time for simulation
    ## use low resolution for testing purpose #src
    if test                                   #src
        dx::Float64 = 0.04                    #src
    else                                      #src
    dx     = 0.005 # grid spacing
    end                                       #src

    ## material constants for grains
    μ = 0.3 # friction coefficient between grains
    elastic = LinearElastic(; E=1e6, ν=0.3)

    ## lattice
    lattice = Lattice(dx, (0,1), (0,1.04))

    #==========
     Particles
    ==========#

    ## grains
    r = 0.06
    if test                                                          #src
        grains::Vector{Marble.infer_particles_type(ParticleState)} = #src
            generate_grains_stable(ParticleState, r, lattice)        #src
    else                                                             #src
    grains = generate_grains(ParticleState, r, lattice)
    end                                                              #src
    for grain in grains
        @. grain.m = 1e3 * grain.V
    end

    ## bar
    if test                                                                                              #src
        bar::Marble.infer_particles_type(ParticleState) =                                                #src
        generate_particles((x,y)->y>1, ParticleState, lattice; alg=PoissonDiskSampling(StableRNG(1234))) #src
    else                                                                                                 #src
    bar = generate_particles((x,y)->y>1, ParticleState, lattice)
    end                                                                                                  #src
    @. bar.v = Vec(0, -0.5)
    @. bar.m = 1 # dammy mass

    @show sum(length, grains)

    #======
     Grids
    ======#

    ## grains
    grain_grids = map(_->generate_grid(GridState, lattice), grains)
    ## bar
    bar_grid = generate_grid(GridState, lattice)
    ## center of mass
    m_cm  = zeros(Float64, size(lattice))
    mv_cm = zeros(Vec{2, Float64}, size(lattice))
    v_cm  = zeros(Vec{2, Float64}, size(lattice))

    #=========
     MPSpaces
    =========#

    grain_spaces = map(grain -> MPSpace(itp, size(lattice), length(grain)), grains)
    bar_space = MPSpace(itp, size(lattice), length(bar))

    #========
     Outputs
    ========#

    if output #src
    pvdfile = joinpath(mkpath("Output.tmp"), "contacting_grains")
    closepvd(openpvd(pvdfile))
    end #src

    t = 0.0
    step = 0
    fps = 50
    savepoints = collect(LinRange(t, t_stop, round(Int, t_stop*fps)+1))
    while t < t_stop

        #===================
         Calculate timestep
        ===================#

        vmax::Float64 = maximum(grains) do grain
            maximum(grain) do pt
                λ, μ = elastic.λ, elastic.μ
                ρ = pt.m / pt.V
                vc = √((λ+2μ) / ρ)
                vc + norm(pt.v)
            end
        end
        Δt = CFL * spacing(lattice) / vmax

        #=============
         P2G transfer
        =============#

        fillzero!.((m_cm, mv_cm))

        ## grains
        Marble.@threaded_inbounds for i in eachindex(grains)
            grid = grain_grids[i]
            grain = grains[i]
            space = grain_spaces[i]
            update!(space, grid, grain; parallel=false)
            particle_to_grid!((:m,:mv,:f,:∇m), fillzero!(grid), grain, space; parallel=false)
        end
        for (grid, grain, space) in zip(grain_grids, grains, grain_spaces)
            @inbounds for i in eachindex(grid)
                if isnonzero(grid, i)
                    m_cm[i] += grid.m[i]
                    mv_cm[i] += grid.mv[i] + Δt * grid.f[i]
                end
            end
            @. grid.vⁿ = grid.mv / grid.m * !iszero(grid.m)
            @. grid.v = grid.vⁿ + Δt*(grid.f/grid.m) * !iszero(grid.m)
        end

        ## bar
        update!(bar_space, bar_grid, bar; parallel=false)
        particle_to_grid!((:m,:mv), fillzero!(bar_grid), bar, bar_space; parallel=false)
        @. bar_grid.v = bar_grid.vⁿ = bar_grid.mv / bar_grid.m * !iszero(bar_grid.m)

        ## center of mass
        @. v_cm = mv_cm / m_cm
        for i in eachindex(bar_grid)
            if !iszero(bar_grid.m[i])
                v_cm[i] = bar_grid.v[i]
            end
        end

        #=======================================
         Impose contact and boundary conditions
        =======================================#

        for (grid, grain) in zip(grain_grids, grains)
            impose_contact_condition!(grid, grain, v_cm, μ)
            impose_boundary_condition!(grid)
        end

        #=============
         G2P transfer
        =============#

        ## grains
        Marble.@threaded_inbounds for i in eachindex(grains)
            grid = grain_grids[i]
            grain = grains[i]
            space = grain_spaces[i]
            grid_to_particle!((:v,:∇v,:x,), grain, grid, space, Δt; parallel=false)
        end
        ## bar
        grid_to_particle!((:x,), bar, bar_grid, bar_space, Δt; parallel=false)

        ## update stress and volume for grains
        Marble.@threaded_inbounds for grain in grains
            for pt in LazyRows(grain)
                ∇v = pt.∇v
                σⁿ = pt.σ
                Δϵ = symmetric(∇v*Δt)
                ΔW = skew(∇v*Δt)

                cᵉ = elastic.c
                σ = (σⁿ + cᵉ ⊡ Δϵ) + symmetric(ΔW⋅σⁿ - σⁿ⋅ΔW)
                pt.σ = σ
                pt.V *= 1 + tr(Δϵ)
            end
        end

        #===================================
         Advance timestep and write results
        ===================================#

        t += Δt
        step += 1

        if output #src
        if t > first(savepoints)
            popfirst!(savepoints)
            openpvd(pvdfile; append=true) do pvd
                openvtm(string(pvdfile, step)) do vtm
                    ## grains
                    for (grain, grid) in zip(grains, grain_grids)
                        openvtk(vtm, grain.x) do vtk
                        end
                    end
                    ## bar
                    openvtk(vtm, bar.x) do vtk
                    end
                    pvd[t] = vtm
                end
            end
        end
        end #src
    end
    reduce(vcat, grains) #src
end

function generate_grains(::Type{ParticleState}, r::Real, lattice::Lattice) where {ParticleState}
    grains = Marble.poisson_disk_sampling((r,1-r), (r,1-r); r=2r)
    map(grains) do centroid
        generate_particles(SphericalDomain(Vec(centroid), r), ParticleState, lattice)
    end
end
function generate_grains_stable(::Type{ParticleState}, r::Real, lattice::Lattice) where {ParticleState}                   #src
    rng = StableRNG(1234)                                                                                           #src
    grains = Marble.PDS.generate(rng, (r,1-r), (r,1-r); r=2r)                                                       #src
    map(grains) do centroid                                                                                         #src
        generate_particles(SphericalDomain(Vec(centroid), r), ParticleState, lattice; alg=PoissonDiskSampling(rng)) #src
    end                                                                                                             #src
end

function impose_contact_condition!(grid::Grid, particles::Particles, v_cm::AbstractArray{<: Vec{2}}, μ::Real)
    @assert size(grid) == size(v_cm)
    xc = sum(particles.m .* particles.x) / sum(particles.m)
    @inbounds for i in eachindex(grid)
        if isnonzero(grid, i) && grid.v[i] != v_cm[i]
            n = normalize(grid.∇m[i])
            vᵢ = grid.v[i]
            vʳ = vᵢ - v_cm[i]
            isincontact = vʳ ⋅ n > 0
            if isincontact
                v̄ₙ = vʳ ⋅ n
                vₜ = vʳ - v̄ₙ*n
                v̄ₜ = norm(vₜ)
                grid.v[i] = vᵢ - (v̄ₙ*n + min(μ*v̄ₙ, v̄ₜ) * vₜ/v̄ₜ)
            end
        end
    end
end

function impose_boundary_condition!(grid::Grid)
    gridindices_floor = @view eachindex(grid)[:, begin]
    gridindices_walls = @view eachindex(grid)[[begin, end],:]
    slip(vᵢ, n) = vᵢ - (vᵢ⋅n)*n
    @inbounds for i in gridindices_floor
        grid.v[i] = slip(grid.v[i], Vec(0,1))
    end
    @inbounds for i in gridindices_walls
        grid.v[i] = slip(grid.v[i], Vec(1,0))
    end
end

## check the result                                                                                                                            #src
using Test                                                                                                                                     #src
@test mean(contacting_grains(QuadraticBSpline(),                   FLIP(); test=true).x) ≈ [0.5036385008537854, 0.19079293004940925] rtol=1e-5 #src
@test mean(contacting_grains(uGIMP(),                              FLIP(); test=true).x) ≈ [0.5035729551556197, 0.19486313891209753] rtol=1e-5 #src
@test mean(contacting_grains(KernelCorrection(QuadraticBSpline()), TPIC(); test=true).x) ≈ [0.5038373871432336, 0.1898585728550991]  rtol=1e-5 #src
