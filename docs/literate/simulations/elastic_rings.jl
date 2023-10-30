# # Elastic rings

using Marble
using StableRNGs #src

include(joinpath(pkgdir(Marble), "docs/literate/models/NeoHookean.jl"))

function elastic_rings(
        itp::Interpolation = KernelCorrection(CubicBSpline()),
        alg::TransferAlgorithm = TPIC();
        implicit::Union{Nothing, TimeIntegrationAlgorithm} = nothing,
        jacobian_free::Bool = true,
        output::Bool = true, #src
        test::Bool = false,  #src
    )

    ## simulation parameters
    if implicit isa TimeIntegrationAlgorithm
        CFL = 4.0 # Courant number
    else
        CFL = 0.8 # Courant number
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
        X  :: Vec{2, Float64}
        m  :: Float64
        mv :: Vec{2, Float64}
        f  :: Vec{2, Float64}
        v  :: Vec{2, Float64}
        vⁿ :: Vec{2, Float64}
        ma :: Vec{2, Float64}
        a  :: Vec{2, Float64}
        aⁿ :: Vec{2, Float64}
        x  :: Vec{2, Float64}
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
        a  :: Vec{2, Float64}
        ∇a :: SecondOrderTensor{2, Float64, 4}
        l  :: Float64                          # for uGIMP
        B  :: SecondOrderTensor{2, Float64, 4} # for APIC
        C  :: Mat{2, 3, Float64, 6}            # for WLS
    end

    ## grid
    grid = generate_grid(GridState, Δx, (-l/2,l/2), (-w/2,w/2))

    ## particles
    if test                                                                                                                                            #src
        particles_lhs = generate_particles((x,y) -> r_in^2 < (x+l/4)^2+y^2 < r_out^2, ParticleState, grid.X; alg=PoissonDiskSampling(StableRNG(1234))) #src
        particles_rhs = generate_particles((x,y) -> r_in^2 < (x-l/4)^2+y^2 < r_out^2, ParticleState, grid.X; alg=PoissonDiskSampling(StableRNG(1234))) #src
    else                                                                                                                                               #src
    particles_lhs = generate_particles((x,y) -> r_in^2 < (x+l/4)^2+y^2 < r_out^2, ParticleState, grid.X)
    particles_rhs = generate_particles((x,y) -> r_in^2 < (x-l/4)^2+y^2 < r_out^2, ParticleState, grid.X)
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
    if implicit isa TimeIntegrationAlgorithm
        integrator = ImplicitIntegrator(implicit, grid, particles; jacobian_free)
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

        if implicit isa NewmarkBeta
            particle_to_grid!(:ma, grid, particles, space; alg)
            @. grid.aⁿ = grid.ma / grid.m * !iszero(grid.m)
        end

        ## boundary conditions
        isfixed = falses(2, size(grid)...)
        @inbounds for i in @view eachindex(grid)[[begin,end],:]
            grid.v[i] = grid.v[i] .* (false,true)
            grid.vⁿ[i] = grid.vⁿ[i] .* (false,true)
            isfixed[1,i] = true
        end

        ## G2P transfer
        function update_stress!(pt, ∇u)
            F = (I + ∇u) ⋅ pt.Fⁿ
            V = det(F) * pt.V⁰
            σ = compute_cauchy_stress(elastic, F)
            pt.F = F
            pt.V = V
            pt.σ = σ
            V * σ
        end
        if implicit isa TimeIntegrationAlgorithm
            solve_grid_velocity!(pt -> gradient(∇u -> update_stress!(pt, ∇u), pt.∇u), grid, particles, space, Δt, integrator; alg, bc=isfixed)
            grid_to_particle!((:v,:∇v,:x), particles, grid, space; alg)
            if implicit isa NewmarkBeta
                grid_to_particle!((:a,:∇a), particles, grid, space; alg)
            end
        else
            grid_to_particle!(pt -> update_stress!(pt, pt.∇v*Δt), (:v,:∇v,:x), particles, grid, space, Δt; alg)
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

## check the result                                                                                                                                                                #src
using Test                                                                                                                                                                         #src
if @isdefined(RUN_TESTS) && RUN_TESTS                                                                                                                                              #src
@test mean(elastic_rings(KernelCorrection(CubicBSpline()), FLIP(); implicit=nothing,         test=true).x) ≈ [-0.0029897053481383294, -0.0001930994229138081] rtol=1e-5            #src
@test mean(elastic_rings(KernelCorrection(CubicBSpline()), APIC(); implicit=nothing,         test=true).x) ≈ [-0.003747610051693535, -0.00019309942291389977] rtol=1e-5            #src
@test mean(elastic_rings(KernelCorrection(CubicBSpline()), TPIC(); implicit=nothing,         test=true).x) ≈ [-0.003708464542420926, -0.00019309942291382]    rtol=1e-5            #src
@test mean(elastic_rings(KernelCorrection(CubicBSpline()), FLIP(); implicit=BackwardEuler(), jacobian_free=true,  test=true).x) ≈                                                  #src
      mean(elastic_rings(KernelCorrection(CubicBSpline()), FLIP(); implicit=BackwardEuler(), jacobian_free=false, test=true).x) ≈ [-0.006999089832465376, -0.0001930994229138141]  #src
@test mean(elastic_rings(KernelCorrection(CubicBSpline()), FLIP(); implicit=NewmarkBeta(),   jacobian_free=true,  test=true).x) ≈                                                  #src
      mean(elastic_rings(KernelCorrection(CubicBSpline()), FLIP(); implicit=NewmarkBeta(),   jacobian_free=false, test=true).x) ≈ [-0.010005398694854659, -0.00019309942291343866] #src
@test mean(elastic_rings(KernelCorrection(CubicBSpline()), APIC(); implicit=BackwardEuler(), jacobian_free=true,  test=true).x) ≈                                                  #src
      mean(elastic_rings(KernelCorrection(CubicBSpline()), APIC(); implicit=BackwardEuler(), jacobian_free=false, test=true).x) ≈ [-0.006870318810593698, -0.00019309942291383616] #src
@test mean(elastic_rings(KernelCorrection(CubicBSpline()), APIC(); implicit=NewmarkBeta(),   jacobian_free=true,  test=true).x) ≈                                                  #src
      mean(elastic_rings(KernelCorrection(CubicBSpline()), APIC(); implicit=NewmarkBeta(),   jacobian_free=false, test=true).x) ≈ [-0.0029948574816484243, -0.00019309942291342]   #src
@test mean(elastic_rings(KernelCorrection(CubicBSpline()), TPIC(); implicit=BackwardEuler(), jacobian_free=true,  test=true).x) ≈                                                  #src
      mean(elastic_rings(KernelCorrection(CubicBSpline()), TPIC(); implicit=BackwardEuler(), jacobian_free=false, test=true).x) ≈ [-0.006868871985344444, -0.0001930994229138209]  #src
@test mean(elastic_rings(KernelCorrection(CubicBSpline()), TPIC(); implicit=NewmarkBeta(),   jacobian_free=true,  test=true).x) ≈                                                  #src
      mean(elastic_rings(KernelCorrection(CubicBSpline()), TPIC(); implicit=NewmarkBeta(),   jacobian_free=false, test=true).x) ≈ [-0.002444840950719716, -0.00019309942291344807] #src
end                                                                                                                                                                                #src
