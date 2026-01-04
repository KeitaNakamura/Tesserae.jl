# # Total Lagrangian MPM
#
# ```@raw html
# <img src="https://github.com/user-attachments/assets/84746086-8456-4bfa-b16f-a0a936fe5472" width="300"/>
# ```
#
# | # Particles | # Iterations | Execution time (w/o output) |
# | ----------- | ------------ | --------------------------- |
# | 8k          | 27k          | 30 sec                      |
#
# This example demonstrates the total lagrangian material point method[^1].
# The implementation solves generalized vortex problem[^1] using a linear kernel.
#
# !!! note
#     Currently, the Bernstein function used in the paper[^1] has not been implemented.
#
# [^1]: [de Vaucorbeil, A., Nguyen, V.P. and Hutchinson, C.R., 2020. A Total-Lagrangian Material Point Method for solid mechanics problems involving large deformations. Computer Methods in Applied Mechanics and Engineering, 360, p.112783.](https://doi.org/10.1016/j.cma.2019.112783)

using Tesserae

function main()

    ## Simulation parameters
    h   = 0.02 # Grid spacing
    T   = 1.0  # Time span
    CFL = 0.1  # Courant number
    α   = 0.99 # PIC-FLIP parameter
    if @isdefined(RUN_TESTS) && RUN_TESTS #src
        h = 0.05                          #src
    end                                   #src

    ## Material constants
    E  = 1e6                    # Young's modulus
    ν  = 0.3                    # Poisson's ratio
    λ  = (E*ν) / ((1+ν)*(1-2ν)) # Lame's first parameter
    μ  = E / 2(1 + ν)           # Shear modulus
    ρ⁰ = 1e3                    # Initial density

    ## Geometry
    Rᵢ = 0.75
    Rₒ = 1.25

    ## Equations for vortex
    G = π
    R̄ = (Rᵢ + Rₒ) / 2
    function calc_b_Rθ(R, t)
        local h′′, h′, h = hessian(R -> 1-8((R-R̄)/(Rᵢ-Rₒ))^2+16((R-R̄)/(Rᵢ-Rₒ))^4, R, :all)
        local g′′, g′, g = hessian(t -> G*sin(π*t/T), t, :all)
        β = g * h
        b_R = ( μ/ρ⁰*(3g*h′+R*g*h′′) - R*g′′*h)*sin(β) + (μ/ρ⁰*R*(g*h′)^2 - R*(g′*h)^2)*cos(β)
        b_θ = (-μ/ρ⁰*(3g*h′+R*g*h′′) + R*g′′*h)*cos(β) + (μ/ρ⁰*R*(g*h′)^2 + R*(g′*h)^2)*sin(β)
        Vec(b_R, b_θ)
    end
    isinside(x::Vec) = Rᵢ^2 < x⋅x < Rₒ^2

    GridProp = @NamedTuple begin
        X    :: Vec{2, Float64}
        m    :: Float64
        m⁻¹  :: Float64
        mv   :: Vec{2, Float64}
        fint :: Vec{2, Float64}
        fext :: Vec{2, Float64}
        b    :: Vec{2, Float64}
        v    :: Vec{2, Float64}
        vⁿ   :: Vec{2, Float64}
    end
    ParticleProp = @NamedTuple begin
        x  :: Vec{2, Float64}
        X  :: Vec{2, Float64}
        m  :: Float64
        V⁰ :: Float64
        v  :: Vec{2, Float64}
        ṽ  :: Vec{2, Float64}
        ã  :: Vec{2, Float64}
        P  :: SecondOrderTensor{2, Float64, 4}
        F  :: SecondOrderTensor{2, Float64, 4}
    end

    ## Background grid
    grid = generate_grid(GridProp, CartesianMesh(h, (-1.5,1.5), (-1.5,1.5)))
    outside_gridinds = findall(!isinside, grid.X)

    ## Particles
    particles = generate_particles(ParticleProp, grid.X; alg=GridSampling(subdiv=1))
    particles.V⁰ .= volume(grid.X) / length(particles)

    filter!(pt->isinside(pt.x), particles)

    @. particles.X = particles.x
    @. particles.m = ρ⁰ * particles.V⁰
    @. particles.F = one(particles.F)
    @show length(particles)

    ## Precompute linear kernel weights
    weights = generate_interpolation_weights(BSpline(Linear()), grid.X, length(particles))
    update!(weights, particles, grid.X)

    ## Outputs
    outdir = mkpath(joinpath("output", "tlmpm_vortex"))
    pvdfile = joinpath(outdir, "paraview")
    closepvd(openpvd(pvdfile)) # Create file

    t = 0.0
    step = 0
    fps = 60
    savepoints = collect(LinRange(t, T, round(Int, T*fps)+1))

    Tesserae.@showprogress while t < T

        ## Calculate time step based on the wave speed
        vmax = maximum(@. sqrt((λ+2μ) / (particles.m/(particles.V⁰ * det(particles.F)))) + norm(particles.v))
        Δt = CFL * h / vmax

        ## Compute grid body forces
        for i in eachindex(grid)
            if isinside(grid.X[i])
                (x, y) = grid.X[i]
                R = sqrt(x^2 + y^2)
                θ = atan(y, x)
                grid.b[i] = rotmat(θ) * calc_b_Rθ(R, t)
            end
        end

        ## Particle-to-grid transfer
        @P2G grid=>i particles=>p weights=>ip begin
            m[i]    = @∑ w[ip] * m[p]
            mv[i]   = @∑ w[ip] * m[p] * v[p]
            fint[i] = @∑ -V⁰[p] * P[p] * ∇w[ip]
            fext[i] = m[i] * b[i]
            m⁻¹[i] = inv(m[i]) * !iszero(m[i])
            vⁿ[i]  = mv[i] * m⁻¹[i]
            v[i]   = vⁿ[i] + ((fint[i] + fext[i]) * m⁻¹[i]) * Δt
        end
        grid.v[outside_gridinds] .= Ref(zero(eltype(grid.v)))

        ## Update particle velocity and position
        @G2P grid=>i particles=>p weights=>ip begin
            ṽ[p]  = @∑ w[ip] * v[i]
            ã[p]  = @∑ w[ip] * (v[i] - vⁿ[i]) / Δt
            v[p]  = (1-α)*ṽ[p] + α*(v[p] + ã[p]*Δt)
            x[p] += ṽ[p] * Δt
        end

        ## Remap updated velocity to grid (MUSL)
        @P2G grid=>i particles=>p weights=>ip begin
            mv[i] = @∑ w[ip] * m[p] * v[p]
            v[i]  = mv[i] * m⁻¹[i]
        end
        grid.v[outside_gridinds] .= Ref(zero(eltype(grid.v)))

        ## Update stress
        @G2P grid=>i particles=>p weights=>ip begin
            F[p] += @∑ Δt * v[i] ⊗ ∇w[ip]
            P[p]  = μ * (F[p] - inv(F[p])') + λ * log(det(F[p])) * inv(F[p])'
        end

        t += Δt
        step += 1

        if t > first(savepoints)
            popfirst!(savepoints)
            openpvd(pvdfile; append=true) do pvd
                openvtm(string(pvdfile, step)) do vtm
                    angle(x) = atan(x[2], x[1])
                    openvtk(vtm, particles.x) do vtk
                        vtk["Velocity (m/s)"] = particles.v
                        vtk["Initial angle (rad)"] = angle.(particles.X)
                    end
                    openvtk(vtm, grid.X) do vtk
                        vtk["External force (N)"] = grid.fext
                    end
                    pvd[t] = vtm
                end
            end
        end
    end
    isapprox(particles.x, particles.X; rtol=0.02) #src
end

using Test                            #src
if @isdefined(RUN_TESTS) && RUN_TESTS #src
    @test main()                      #src
end                                   #src
