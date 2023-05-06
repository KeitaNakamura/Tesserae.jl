# # Sand column collapse
#
# ```@raw html
# <img class="display-light-only" src="https://user-images.githubusercontent.com/16015926/233009892-d8d9265c-2b99-493b-943f-0246118cb5ec.mp4"/>
# <img class="display-dark-only" src="https://user-images.githubusercontent.com/16015926/233009969-ded82325-0a62-4a49-89a7-bcf627d11fe2.mp4"/>
# ```

using Marble
using StableRNGs #src

include(joinpath(pkgdir(Marble), "docs/literate/models/DruckerPrager.jl"))

function sand_column_collapse(
        itp::Interpolation = QuadraticBSpline(),
        alg::TransferAlgorithm = FLIP(),
        ;output::Bool = true, #src
        test::Bool = false,   #src
    )

    ## simulation parameters
    g      = 9.81 # gravity acceleration
    CFL    = 1.0  # Courant number
    t_stop = 1.4  # time for simulation
    ## use low resolution for testing purpose #src
    if test                                   #src
        Δx::Float64 = 0.05                    #src
    else                                      #src
    Δx     = 0.01 # grid spacing
    end                                       #src

    ## material constants for soil (Drucker-Prager model with linear elastic model)
    ρ₀      = 1.5e3 # initial density
    elastic = LinearElastic(; E=1e6, ν=0.3)
    model   = DruckerPrager(:plane_strain, elastic; c=0.0, ϕ=deg2rad(35), ψ=deg2rad(0))

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
        v  :: Vec{2, Float64}
        ∇v :: SecondOrderTensor{3, Float64, 9}
        σ  :: SymmetricSecondOrderTensor{3, Float64, 6}
        b  :: Vec{2, Float64}
        l  :: Float64                          # for uGIMP
        B  :: SecondOrderTensor{2, Float64, 4} # for APIC
        C  :: Mat{2, 3, Float64, 6}            # for WLS
    end

    ## grid
    grid = generate_grid(GridState, Δx, (-1.5,1.5), (0,1))

    ## particles
    h = 0.9 # height of sand column
    if test                                                                                                                 #src
        particles::Marble.infer_particles_type(ParticleState) =                                                             #src
            generate_particles((x,y) -> -0.3<x<0.3 && y<h, ParticleState, grid.x; alg=PoissonDiskSampling(StableRNG(1234))) #src
    else                                                                                                                    #src
    particles = generate_particles((x,y) -> -0.3<x<0.3 && y<h, ParticleState, grid.x)
    end                                                                                                                     #src
    for pt in LazyRows(particles)
        ν = elastic.ν
        y = pt.x[2]
        σ_y = -ρ₀ * g * (h-y)
        σ_x = σ_y * ν / (1-ν)
        pt.σ = symmetric(@Mat [σ_x 0   0
                               0   σ_y 0
                               0   0   σ_x])
    end
    @. particles.m = ρ₀ * particles.V
    @. particles.b = Vec(0, -g)
    @show length(particles)

    ## create interpolation space
    space = MPSpace(itp, size(grid), length(particles))

    ## outputs
    if output                                               #src
    outdir = joinpath("output.tmp", "sand_column_collapse")
    rm(outdir; recursive=true, force=true)                  #src
    pvdfile = joinpath(mkpath(outdir), "paraview")
    closepvd(openpvd(pvdfile))
    end                                                     #src

    t = 0.0
    step = 0
    fps = 50
    savepoints = collect(LinRange(t, t_stop, round(Int, t_stop*fps)+1))
    while t < t_stop

        ## calculate timestep based on the Courant-Friedrichs-Lewy (CFL) condition
        Δt = CFL * spacing(grid) / maximum(LazyRows(particles)) do pt
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
        @. grid.vⁿ = grid.mv / grid.m
        @. grid.v = grid.vⁿ + Δt*(grid.f/grid.m)

        ## boundary conditions

        ## consider friction on the floor
        gridindices_floor = @view eachindex(grid)[:, begin]
        @inbounds for i in gridindices_floor
            μ = 0.4 # friction coefficient
            n = Vec(0,-1)
            vᵢ = grid.v[i]
            if !iszero(vᵢ)
                v̄ₙ = vᵢ ⋅ n
                vₜ = vᵢ - v̄ₙ*n
                v̄ₜ = norm(vₜ)
                grid.v[i] = vᵢ - (v̄ₙ*n + min(μ*v̄ₙ, v̄ₜ) * vₜ/v̄ₜ)
            end
        end

        ## slip condition on the walls
        gridindices_walls = @view eachindex(grid)[[begin, end],:]
        @inbounds for i in gridindices_walls
            n = Vec(1,0) # this is ok for left side as well
            vᵢ = grid.v[i]
            grid.v[i] = vᵢ - (vᵢ⋅n)*n
        end

        ## G2P transfer
        grid_to_particle!((:v,:∇v,:x), particles, grid, space, Δt; alg)

        ## update other particle states
        Marble.@threads_inbounds for pt in LazyRows(particles)
            ∇v = pt.∇v
            σⁿ = pt.σ
            Δϵ = symmetric(∇v*Δt)
            ΔW = skew(∇v*Δt)

            σ = compute_stress(model, σⁿ, Δϵ)
            σ = σ + symmetric(ΔW⋅σⁿ - σⁿ⋅ΔW)
            pt.σ = σ
            pt.V *= 1 + tr(Δϵ)
        end

        t += Δt
        step += 1

        if output #src
        if t > first(savepoints)
            popfirst!(savepoints)
            openpvd(pvdfile; append=true) do pvd
                openvtk(string(pvdfile, step), particles.x) do vtk
                    vtk["velocity"] = particles.v
                    pvd[t] = vtk
                end
            end
        end
        end #src
    end
    particles #src
end

## check the result                                                                                                                                  #src
using Test                                                                                                                                           #src
if @isdefined(RUN_TESTS) && RUN_TESTS                                                                                                                #src
@test mean(sand_column_collapse(QuadraticBSpline(),                   FLIP(); test=true).x) ≈ [-0.005200135291629036, 0.13121478975513662] rtol=1e-5 #src
@test mean(sand_column_collapse(QuadraticBSpline(),               FLIP(0.95); test=true).x) ≈ [-0.005674765938663493, 0.14500628912686417] rtol=1e-5 #src
@test mean(sand_column_collapse(uGIMP(),                              FLIP(); test=true).x) ≈ [-0.00730333735180601, 0.13718864185942603]  rtol=1e-5 #src
@test mean(sand_column_collapse(KernelCorrection(QuadraticBSpline()), TPIC(); test=true).x) ≈ [-0.007105884808737926, 0.13034912708862376] rtol=1e-5 #src
@test mean(sand_column_collapse(KernelCorrection(QuadraticBSpline()), APIC(); test=true).x) ≈ [-0.006958826887434974, 0.13029970110634176] rtol=1e-5 #src
@test mean(sand_column_collapse(LinearWLS(QuadraticBSpline()),        TPIC(); test=true).x) ≈ [-0.008313504889429786, 0.1328920134070544]  rtol=1e-5 #src
@test mean(sand_column_collapse(LinearWLS(QuadraticBSpline()), WLSTransfer(); test=true).x) ≈ [-0.008313504889429979, 0.1328920134070544]  rtol=1e-5 #src
end                                                                                                                                                  #src
