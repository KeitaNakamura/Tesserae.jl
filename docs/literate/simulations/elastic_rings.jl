# # Elastic rings

using Marble
using StableRNGs #src

include(joinpath(pkgdir(Marble), "docs/literate/models/NeoHookean.jl"))

function elastic_rings(
        itp::Interpolation = KernelCorrection(QuadraticBSpline()),
        alg::TransferAlgorithm = TPIC(),
        ;output::Bool = true, #src
        test::Bool = false,   #src
    )

    ## simulation parameters
    CFL    = 1.0   # Courant number
    t_stop = 10e-3 # time for simulation
    ## use low resolution for testing purpose #src
    if test                                   #src
        Δx::Float64 = 1.0e-3                  #src
    else                                      #src
    Δx     = 0.625e-3 # grid spacing
    end                                       #src
    v₀ = Vec(30, 0) # m/s
    w = 150e-3
    l = 200e-3

    ## material constants
    ρ₀      = 1.01e3 # initial density
    r_in    = 30e-3
    r_out   = 40e-3
    elastic = NeoHookean(; E=73.08e6, ν=0.4)

    ## states for grid and particles
    GridState = @NamedTuple begin
        x  :: Vec{2, Float64}
        m  :: Float64
        mv :: Vec{2, Float64}
        f  :: Vec{2, Float64}
        v  :: Vec{2, Float64}
        vⁿ :: Vec{2, Float64}
        bc :: Vec{2, Bool}
    end
    ParticleState = @NamedTuple begin
        x  :: Vec{2, Float64}
        m  :: Float64
        V  :: Float64
        V₀ :: Float64
        v  :: Vec{2, Float64}
        ∇v :: SecondOrderTensor{3, Float64, 9}
        σ  :: SymmetricSecondOrderTensor{3, Float64, 6}
        F  :: SecondOrderTensor{3, Float64, 9}
        l  :: Float64                          # for uGIMP
        B  :: SecondOrderTensor{2, Float64, 4} # for APIC
        C  :: Mat{2, 3, Float64, 6}            # for WLS
    end

    ## grid
    grid = generate_grid(GridState, Δx, (-l/2,l/2), (-w/2,w/2))

    ## particles
    if test                                                                                                                       #src
        particles_lhs = generate_particles((x,y) -> r_in^2 < (x+l/4)^2+y^2 < r_out^2, ParticleState, grid.x; alg=StableRNG(1234)) #src
        particles_rhs = generate_particles((x,y) -> r_in^2 < (x-l/4)^2+y^2 < r_out^2, ParticleState, grid.x; alg=StableRNG(1234)) #src
    else                                                                                                                          #src
    particles_lhs = generate_particles((x,y) -> r_in^2 < (x+l/4)^2+y^2 < r_out^2, ParticleState, grid.x)
    particles_rhs = generate_particles((x,y) -> r_in^2 < (x-l/4)^2+y^2 < r_out^2, ParticleState, grid.x)
    end                                                                                                                        #src
    @. particles_lhs.v =  v₀
    @. particles_rhs.v = -v₀
    particles = [particles_lhs; particles_rhs]
    @. particles.V₀ = particles.V
    @. particles.m  = ρ₀ * particles.V
    @. particles.F  = one(particles.F)
    @show length(particles)

    ## create interpolation space
    space = MPSpace(itp, size(grid), length(particles))

    ## boundary conditions
    fixedbc = fill(false, 2, size(grid)...)
    @inbounds for i in @view eachindex(grid)[[begin,end],:]
        fixedbc[1,i] = true
    end
    fixedbcindices = findall(fixedbc)

    ## outputs
    if output                                                #src
    outdir = joinpath("output.tmp", "elastic_rings")
    rm(outdir; recursive=true, force=true)                   #src
    pvdfile = joinpath(mkpath(outdir), "paraview")
    closepvd(openpvd(pvdfile))
    end                                                      #src

    t = 0.0
    step = 0
    fps = 20e3
    savepoints = collect(LinRange(t, t_stop, round(Int, t_stop*fps)+1))
    Marble.@showprogress while t < t_stop

        ## calculate timestep based on the Courant-Friedrichs-Lewy (CFL) condition
        Δt = CFL * spacing(grid) / maximum(eachparticle(particles)) do pt
            λ, μ = elastic.λ, elastic.μ
            ρ = pt.m / pt.V
            vc = √((λ+2μ) / ρ)
            vc + norm(pt.v)
        end

        ## update interpolation space
        update!(space, grid, particles)

        ## P2G transfer
        particle_to_grid!((:m,:mv,:f), fillzero!(grid), particles, space; alg)

        ## solve momentum equation
        @. grid.vⁿ = grid.mv / grid.m * !iszero(grid.m)
        @. grid.v = grid.vⁿ + Δt*(grid.f/grid.m) * !iszero(grid.m)

        ## boundary conditions
        flatarray(grid.v)[fixedbcindices] .= false

        ## G2P transfer
        grid_to_particle!((:v,:∇v,:x), particles, grid, space, Δt; alg)

        ## update other particle states
        Marble.@threads_inbounds for pt in eachparticle(particles)
            F = (I + Δt*pt.∇v) ⋅ pt.F
            pt.σ = compute_cauchy_stress(elastic, F)
            pt.F = F
            pt.V = det(F) * pt.V₀
        end

        t += Δt
        step += 1

        if output #src
        if t > first(savepoints)
            popfirst!(savepoints)
            openpvd(pvdfile; append=true) do pvd
                openvtk(string(pvdfile, step), particles.x) do vtk
                    vtk["velocity"] = particles.v
                    vtk["von Mises"] = vonmises.(particles.σ)
                    pvd[t] = vtk
                end
            end
        end
        end #src
    end
    ifelse(test, particles, nothing) #src
end

## check the result                                                                                                                                                       #src
using Test                                                                                                                                                                #src
if @isdefined(RUN_TESTS) && RUN_TESTS                                                                                                                                     #src
# @test mean(elastic_rings(KernelCorrection(QuadraticBSpline()), TPIC(); test=true).x) ≈ [-0.004382367540378365, 2.443396567204942, -0.13044416953356866] rtol=1e-5 #src
end                                                                                                                                                                       #src
