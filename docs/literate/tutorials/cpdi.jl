# # A convected particle domain interpolation technique
#
# ```@raw html
# <img src="https://github.com/user-attachments/assets/0052df2b-f532-45d8-97bf-a2967c6b3f0d" width="210"/>
# ```
#
# | # Particles | # Iterations | Execution time |
# | ----------- | ------------ | -------------- |
# | 27k         | 500          | 30 sec         |
#
# This example employs a convected particle domain interpolation[^1].
#
# [^1]: [Sadeghirad, A., Brannon, R.M. and Burghardt, J., 2011. A convected particle domain interpolation technique to extend applicability of the material point method for problems involving massive deformations. International Journal for numerical methods in Engineering, 86(12), pp.1435-1456.](https://doi.org/10.1002/nme.3110)

using Tesserae

function main()

    ## Simulation parameters
    h  = 0.1    # Grid spacing
    T  = 0.5    # Time span
    Δt = 0.001  # Time step
    g  = 1000.0 # Gravity acceleration
    if @isdefined(RUN_TESTS) && RUN_TESTS #src
        T = 0.2                           #src
    end                                   #src

    ## Material constants
    E = 1e6                     # Young's modulus
    ν = 0.3                     # Poisson's ratio
    λ  = (E*ν) / ((1+ν)*(1-2ν)) # Lame's first parameter
    μ  = E / 2(1 + ν)           # Shear modulus
    ρ⁰ = 1050.0                 # Initial density

    GridProp = @NamedTuple begin
        x   :: Vec{3, Float64}
        m   :: Float64
        m⁻¹ :: Float64
        mv  :: Vec{3, Float64}
        f   :: Vec{3, Float64}
        v   :: Vec{3, Float64}
        vⁿ  :: Vec{3, Float64}
    end
    ParticleProp = @NamedTuple begin
        x  :: Vec{3, Float64}
        m  :: Float64
        V⁰ :: Float64
        v  :: Vec{3, Float64}
        ∇v :: SecondOrderTensor{3, Float64, 9}
        σ  :: SymmetricSecondOrderTensor{3, Float64, 6}
        ## Required in CPDI()
        F  :: SecondOrderTensor{3, Float64, 9}
        l  :: Float64
    end

    ## Background grid
    grid = generate_grid(GridProp, CartesianMesh(h, (-1,1), (-3,0.5), (-1,1)))

    ## Particles
    bar = extract(grid.x, (-0.5,0.5), (-1,0), (-0.5,0.5))
    particles = generate_particles(ParticleProp, bar; alg=GridSampling(spacing=1/3))
    particles.V⁰ .= volume(bar) / length(particles)
    @. particles.m = ρ⁰ * particles.V⁰
    @. particles.l = (particles.V⁰)^(1/3)
    @. particles.F = one(particles.F)
    @show length(particles)

    ## Interpolation
    mpvalues = generate_mpvalues(CPDI(), grid.x, length(particles))

    ## Material model (neo-Hookean)
    function cauchy_stress(F)
        J = det(F)
        b = symmetric(F ⋅ F')
        (μ*(b-I) + λ*log(J)*I) / J
    end

    ## Outputs
    outdir = mkpath(joinpath("output", "cpdi"))
    pvdfile = joinpath(outdir, "paraview")
    closepvd(openpvd(pvdfile)) # Create file

    t = 0.0
    step = 0
    fps = 240
    savepoints = collect(LinRange(t, T, round(Int, T*fps)+1))

    Tesserae.@showprogress while t < T

        ## Update interpolation values
        for p in eachindex(particles, mpvalues)
            update!(mpvalues[p], particles[p], grid.x)
        end

        ## Particle-to-grid transfer
        @P2G grid=>i particles=>p mpvalues=>ip begin
            m[i]  = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
            f[i]  = @∑ -V⁰[p] * det(F[p]) * σ[p] ⋅ ∇w[ip] + w[ip] * m[p] * Vec(0,-g,0)
        end

        ## Update grid velocity
        @. grid.m⁻¹ = inv(grid.m) * !iszero(grid.m)
        @. grid.vⁿ = grid.mv * grid.m⁻¹
        @. grid.v  = grid.vⁿ + (grid.f * grid.m⁻¹) * Δt

        ## Boundary conditions
        for i in eachindex(grid)
            if grid.x[i][2] ≥ 0
                grid.vⁿ[i] = grid.vⁿ[i] .* (true,false,true)
                grid.v[i] = grid.v[i] .* (true,false,true)
            end
        end

        ## Grid-to-particle transfer
        @G2P grid=>i particles=>p mpvalues=>ip begin
            v[p]  += @∑ w[ip] * (v[i] - vⁿ[i])
            ∇v[p]  = @∑ v[i] ⊗ ∇w[ip]
            x[p]  += @∑ w[ip] * v[i] * Δt
        end

        ## Update other particle properties
        for p in eachindex(particles)
            ∇uₚ = particles.∇v[p] * Δt
            Fₚ = (I + ∇uₚ) ⋅ particles.F[p]
            σₚ = cauchy_stress(Fₚ)
            particles.σ[p] = σₚ
            particles.F[p] = Fₚ
        end

        t += Δt
        step += 1

        if t > first(savepoints)
            popfirst!(savepoints)
            openpvd(pvdfile; append=true) do pvd
                openvtm(string(pvdfile, step)) do vtm
                    openvtk(vtm, particles.x) do vtk
                        vtk["Velocity (m/s)"] = particles.v
                    end
                    openvtk(vtm, grid.x) do vtk
                    end
                    pvd[t] = vtm
                end
            end
        end
    end
    mean(particles.x) #src
end

using Test                                 #src
if @isdefined(RUN_TESTS) && RUN_TESTS      #src
    @test main() ≈ [0,-0.6154,0] rtol=1e-4 #src
end                                        #src
