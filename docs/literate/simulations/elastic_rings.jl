# # Elastic rings

using Marble
using StableRNGs #src

include(joinpath(pkgdir(Marble), "docs/literate/models/NeoHookean.jl"))

function elastic_rings(
        itp::Interpolation = KernelCorrection(CubicBSpline()),
        alg::TransferAlgorithm = TPIC();
        implicit::Bool = true,
        output::Bool = true, #src
        test::Bool = false,  #src
    )

    ## simulation parameters
    if implicit
        CFL = 4.0 # Courant number
    else
        CFL = 1.0 # Courant number
    end
    t_stop = 10e-3 # time for simulation
    ## use low resolution for testing purpose #src
    if test                                   #src
        Δx::Float64 = 5.0e-3                  #src
    else                                      #src
    Δx = 0.625e-3 # grid spacing
    end                                       #src
    v⁰ = Vec(30, 0) # m/s
    w = 150e-3
    l = 200e-3

    ## material constants
    ρ⁰      = 1.01e3 # initial density
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
    end
    ParticleState = @NamedTuple begin
        x  :: Vec{2, Float64}
        m  :: Float64
        V  :: Float64
        V⁰ :: Float64
        v  :: Vec{2, Float64}
        ∇v :: SecondOrderTensor{3, Float64, 9}
        σ  :: SymmetricSecondOrderTensor{3, Float64, 6}
        F  :: SecondOrderTensor{3, Float64, 9}
        Fⁿ :: SecondOrderTensor{3, Float64, 9}
        l  :: Float64                          # for uGIMP
        B  :: SecondOrderTensor{2, Float64, 4} # for APIC
        C  :: Mat{2, 3, Float64, 6}            # for WLS
    end

    ## grid
    grid = generate_grid(GridState, Δx, (-l/2,l/2), (-w/2,w/2))

    ## particles
    if test                                                                                                                                            #src
        particles_lhs = generate_particles((x,y) -> r_in^2 < (x+l/4)^2+y^2 < r_out^2, ParticleState, grid.x; alg=PoissonDiskSampling(StableRNG(1234))) #src
        particles_rhs = generate_particles((x,y) -> r_in^2 < (x-l/4)^2+y^2 < r_out^2, ParticleState, grid.x; alg=PoissonDiskSampling(StableRNG(1234))) #src
    else                                                                                                                                               #src
    particles_lhs = generate_particles((x,y) -> r_in^2 < (x+l/4)^2+y^2 < r_out^2, ParticleState, grid.x)
    particles_rhs = generate_particles((x,y) -> r_in^2 < (x-l/4)^2+y^2 < r_out^2, ParticleState, grid.x)
    end                                                                                                                                                #src
    @. particles_lhs.v =  v⁰
    @. particles_rhs.v = -v⁰
    particles = [particles_lhs; particles_rhs]
    @. particles.V⁰ = particles.V
    @. particles.m  = ρ⁰ * particles.V
    @. particles.F  = one(particles.F)
    @. particles.Fⁿ = one(particles.Fⁿ)
    @show length(particles)

    ## create interpolation space
    space = MPSpace(itp, size(grid), length(particles))

    ## implicit method
    if implicit
        solver = EulerIntegrator(grid, particles)
    end

    ## outputs
    if output                                        #src
    outdir = joinpath("output.tmp", "elastic_rings")
    rm(outdir; recursive=true, force=true)           #src
    pvdfile = joinpath(mkpath(outdir), "paraview")
    closepvd(openpvd(pvdfile))
    end                                              #src

    t = 0.0
    step = 0
    fps = 20e3
    savepoints = collect(LinRange(t, t_stop, round(Int, t_stop*fps)+1))
    while t < t_stop

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
        isfixed = falses(2, size(grid)...)
        @inbounds for i in @view eachindex(grid)[[begin,end],:]
            grid.v[i] = grid.v[i] .* (false,true)
            grid.vⁿ[i] = grid.vⁿ[i] .* (false,true)
            isfixed[1,i] = true
        end

        ## implicit G2P transfer
        if implicit
            grid_to_particle!((:v,:∇v,:x), particles, grid, space, Δt, solver; alg, bc=isfixed) do pt
                gradient(pt.∇v) do ∇v
                    F = (I + Δt*∇v) ⋅ pt.Fⁿ
                    V = det(F) * pt.V⁰
                    σ = compute_cauchy_stress(elastic, F)
                    pt.F = F
                    pt.V = V
                    pt.σ = σ
                    V * σ
                end
            end
        else
            grid_to_particle!((:v,:∇v,:x), particles, grid, space, Δt; alg) do pt
                F = (I + Δt*pt.∇v) ⋅ pt.Fⁿ
                V = det(F) * pt.V⁰
                σ = compute_cauchy_stress(elastic, F)
                pt.F = F
                pt.V = V
                pt.σ = σ
            end
        end
        @. particles.Fⁿ = particles.F

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

## check the result                                                                                                                                           #src
using Test                                                                                                                                                    #src
if @isdefined(RUN_TESTS) && RUN_TESTS                                                                                                                         #src
@test mean(elastic_rings(KernelCorrection(CubicBSpline()), FLIP(); implicit=false, test=true).x) ≈ [-0.003070200867754963, -0.00019309942291382027] rtol=1e-5 #src
@test mean(elastic_rings(KernelCorrection(CubicBSpline()), APIC(); implicit=false, test=true).x) ≈ [-0.0038299836923257747, -0.0001930994229137899] rtol=1e-5 #src
@test mean(elastic_rings(KernelCorrection(CubicBSpline()), TPIC(); implicit=false, test=true).x) ≈ [-0.0037910149463340173, -0.0001930994229138152] rtol=1e-5 #src
@test mean(elastic_rings(KernelCorrection(CubicBSpline()), FLIP(); implicit=true,  test=true).x) ≈ [-0.006999089832465376, -0.0001930994229138141]  rtol=1e-5 #src
@test mean(elastic_rings(KernelCorrection(CubicBSpline()), APIC(); implicit=true,  test=true).x) ≈ [-0.006870318810593698, -0.00019309942291383616] rtol=1e-5 #src
@test mean(elastic_rings(KernelCorrection(CubicBSpline()), TPIC(); implicit=true,  test=true).x) ≈ [-0.006868871985344444, -0.0001930994229138209]  rtol=1e-5 #src
end                                                                                                                                                           #src
