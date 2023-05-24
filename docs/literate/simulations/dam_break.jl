# # Dam-break
#
# ```@raw html
# <img class="display-light-only" src="https://user-images.githubusercontent.com/16015926/232998791-c4347e90-4c55-4057-8a2d-5c681e5fbc65.mp4" alt="dam_break"/>
# <img class="display-dark-only" src="https://user-images.githubusercontent.com/16015926/232999036-d5cef186-702e-4230-b403-059137cdebeb.mp4" alt="dam_break"/>
# ```

using Marble
using StableRNGs #src

function dam_break(
        itp::Interpolation = KernelCorrection(QuadraticBSpline()),
        alg::TransferAlgorithm = TPIC(),
        ;output::Bool = true, #src
        test::Bool = false,   #src
    )

    ## simulation parameters
    t_stop = 3.0   # time for simulation
    g      = 9.81  # gravity acceleration
    CFL    = 0.1   # Courant number
    ## use low resolution for testing purpose #src
    if test                                   #src
        Δx::Float64 = 0.07                    #src
    else                                      #src
    Δx     = 0.014 # grid spacing
    end                                       #src

    ## material constants for water
    ρ₀ = 1.0e3   # initial density
    μ  = 1.01e-3 # dynamic viscosity (Pa⋅s)
    c  = 60.0    # speed of sound (m/s)

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
    grid = generate_grid(GridState, Δx, (0,3.22), (0,4.0))

    ## particles
    if test                                                                                                              #src
        particles::Marble.infer_particles_type(ParticleState) =                                                          #src
            generate_particles((x,y) -> x<1.2 && y<0.6, ParticleState, grid.x; alg=PoissonDiskSampling(StableRNG(1234))) #src
    else                                                                                                                 #src
    particles = generate_particles((x,y) -> x<1.2 && y<0.6, ParticleState, grid.x)
    end                                                                                                                  #src
    @. particles.m = ρ₀ * particles.V
    @. particles.b = Vec(0, -g)
    @show length(particles)

    ## create interpolation space
    space = MPSpace(itp, size(grid), length(particles))

    ## outputs
    if output                                      #src
    outdir = joinpath("output.tmp", "dam_break")
    rm(outdir; recursive=true, force=true)         #src
    pvdfile = joinpath(mkpath(outdir), "paraview")
    closepvd(openpvd(pvdfile))
    end                                            #src

    t = 0.0
    step = 0
    fps = 50
    savepoints = collect(LinRange(t, t_stop, round(Int, t_stop*fps)+1))
    while t < t_stop

        ## calculate timestep based on the Courant-Friedrichs-Lewy (CFL) condition
        Δt = CFL * minimum(eachparticle(particles)) do pt
            ρ = pt.m / pt.V
            ν = μ / ρ # kinemtatic viscosity
            min(Δx/(c+norm(pt.v)), Δx^2/ν)
        end

        ## update interpolation space
        update!(space, grid, particles)

        ## P2G transfer
        particle_to_grid!((:m,:mv,:f), fillzero!(grid), particles, space; alg)

        ## solve momentum equation
        @. grid.vⁿ = grid.mv / grid.m * !iszero(grid.m)
        @. grid.v = grid.vⁿ + Δt*(grid.f/grid.m) * !iszero(grid.m)

        ## boundary conditions
        @inbounds for i in @view eachindex(grid)[:,begin] # floor
            grid.v[i] = grid.v[i] .* (true,false)
        end
        @inbounds for i in @view eachindex(grid)[[begin,end],:] # walls
            grid.v[i] = grid.v[i] .* (false,true)
        end

        ## G2P transfer
        grid_to_particle!((:v,:∇v,:x), particles, grid, space, Δt; alg)

        ## update other particle states
        Marble.@threads_inbounds for pt in eachparticle(particles)
            d = symmetric(pt.∇v)
            V = pt.V * exp(tr(d)*Δt)
            ρ = pt.m / V
            p = c^2 * (ρ - ρ₀)
            pt.σ = -p*I + 2μ*dev(d)
            pt.V = V
        end

        t += Δt
        step += 1

        if output #src
        if t > first(savepoints)
            popfirst!(savepoints)
            openpvd(pvdfile; append=true) do pvd
                openvtk(string(pvdfile, step), particles.x) do vtk
                    vorticity(∇v) = ∇v[2,1] - ∇v[1,2]
                    vtk["vorticity"] = @. vorticity(particles.∇v)
                    pvd[t] = vtk
                end
            end
        end
        end #src
    end
    ifelse(test, particles, nothing) #src
end

## check the result                                                                                                                           #src
using Test                                                                                                                                    #src
if @isdefined(RUN_TESTS) && RUN_TESTS                                                                                                         #src
@test mean(dam_break(KernelCorrection(QuadraticBSpline()), TPIC();        test=true).x) ≈ [1.626196432341774, 0.11209068688080602]  rtol=1e-5 #src
@test mean(dam_break(KernelCorrection(QuadraticBSpline()), APIC();        test=true).x) ≈ [1.6265820036691907, 0.11217602693564092] rtol=1e-5 #src
@test mean(dam_break(KernelCorrection(QuadraticBSpline()), FLIP();        test=true).x) ≈ [1.5109943159336998, 0.12369021905817391] rtol=1e-5 #src
@test mean(dam_break(KernelCorrection(QuadraticBSpline()), FLIP(0.95);    test=true).x) ≈ [1.7229957080150673, 0.11395714240142928] rtol=1e-5 #src
@test mean(dam_break(LinearWLS(QuadraticBSpline()),        TPIC();        test=true).x) ≈                                                     #src
      mean(dam_break(LinearWLS(QuadraticBSpline()),        WLSTransfer(); test=true).x)                                                       #src
end                                                                                                                                           #src
