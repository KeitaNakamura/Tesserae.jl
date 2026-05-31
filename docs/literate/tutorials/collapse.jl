# # Elasto-plastic large deformation
#
# ```@raw html
# <img src="https://github.com/user-attachments/assets/16a48085-6977-4c6f-b9fa-a9634e73eeef" width="800"/>
# ```
#
# | # Particles | # Iterations | Execution time (w/o output) |
# | ----------- | ------------ | ----------------------------|
# | 19k         | 8k           | 20 sec                      |
#
# ## Drucker--Prager model

using Tesserae

@kwdef struct DruckerPrager
    λ  :: Float64            # Lame's first parameter
    G  :: Float64            # Shear modulus
    ϕ  :: Float64            # Internal friction angle
    ψ  :: Float64 = ϕ        # Dilatancy angle
    c  :: Float64 = 0.0      # Cohesion
    pₜ :: Float64 = c/tan(ϕ) # Mean stress for tension limit
    ## Assume plane strain condition
    A  :: Float64 = 3√2c      / sqrt(9+12tan(ϕ)^2)
    B  :: Float64 = 3√2tan(ϕ) / sqrt(9+12tan(ϕ)^2)
    b  :: Float64 = 3√2tan(ψ) / sqrt(9+12tan(ψ)^2)
end

function cauchy_stress(model::DruckerPrager, σⁿ::SymmetricSecondOrderTensor{3}, ∇u::SecondOrderTensor{3})
    δ = one(SymmetricSecondOrderTensor{3})
    I = one(SymmetricFourthOrderTensor{3})

    (; λ, G, A, B, b, pₜ) = model

    f(σ) = norm(dev(σ)) - (A - B*tr(σ)/3) # Yield function
    g(σ) = norm(dev(σ)) + b*tr(σ)/3       # Plastic potential function

    ## Elastic predictor
    cᵉ = λ*δ⊗δ + 2G*I
    σᵗʳ = σⁿ + cᵉ ⊡₂ symmetric(∇u) + 2*symmetric(σⁿ * skew(∇u)) # Consider Jaumann stress-rate
    dfdσ, fᵗʳ = gradient(f, σᵗʳ, :all)
    fᵗʳ ≤ 0 && tr(σᵗʳ)/3 ≤ pₜ && return σᵗʳ

    ## Plastic corrector
    dgdσ = gradient(g, σᵗʳ)
    Δλ = fᵗʳ / (dfdσ ⊡₂ cᵉ ⊡₂ dgdσ)
    Δεᵖ = Δλ * dgdσ
    σ = σᵗʳ - cᵉ ⊡₂ Δεᵖ

    ## Simple tension cutoff
    if !(tr(σ)/3 ≤ pₜ) # σᵗʳ is not in zone1
        ##
        ## \<- yield surface
        ##  \         /
        ##   \ zone1 /
        ##    \     /   zone2
        ##     \   /
        ##      \ /______________
        ##       |
        ##       |      zone3
        ##       |
        ## ------------------------> p
        ##       pₜ
        ##
        s = dev(σᵗʳ)
        σ = pₜ*δ + s
        if f(σ) > 0 # σ is in zone2
            ## Map to corner
            p = tr(σ) / 3
            σ = pₜ*δ + (A-B*p)*normalize(s)
        end
    end

    σ
end

# ## Sand column collapse

function main()

    ## Simulation parameters
    h   = 0.01 # Grid spacing
    t_stop = 1.5 # Final time
    g   = 9.81 # Gravity acceleration
    CFL = 1.0  # Courant number
    if @isdefined(RUN_TESTS) && RUN_TESTS #src
        h = 0.015                         #src
        t_stop = 0.75                     #src
    end                                   #src

    ## Material constants
    E  = 1e6                    # Young's modulus
    ν  = 0.3                    # Poisson's ratio
    λ  = (E*ν) / ((1+ν)*(1-2ν)) # Lame's first parameter
    G  = E / 2(1 + ν)           # Shear modulus
    ϕ  = deg2rad(32)            # Internal friction angle
    ψ  = deg2rad(0)             # Dilatancy angle
    ρ⁰ = 1.5e3                  # Initial density

    ## Geometry
    H = 0.8 # Height of sand column
    W = 0.6 # Width of sand column

    GridProp = @NamedTuple begin
        x  :: Vec{2, Float64}
        m  :: Float64
        m⁻¹ :: Float64
        v  :: Vec{2, Float64}
        vⁿ :: Vec{2, Float64}
        mv :: Vec{2, Float64}
        f  :: Vec{2, Float64}
    end
    ParticleProp = @NamedTuple begin
        x  :: Vec{2, Float64}
        m  :: Float64
        V  :: Float64
        v  :: Vec{2, Float64}
        ∇v :: SecondOrderTensor{2, Float64, 4}
        F  :: SecondOrderTensor{2, Float64, 4}
        σ  :: SymmetricSecondOrderTensor{3, Float64, 6}
    end

    ## Background grid
    grid = generate_grid(GridProp, CartesianMesh(h, (-3,3), (0,1)))

    ## Particles
    particles = generate_particles(ParticleProp, grid.x)
    particles.V .= volume(grid.x) / length(particles)
    filter!(particles) do pt
        x, y = pt.x
        -W/2 < x < W/2 && y < H
    end
    for p in eachindex(particles)
        y = particles.x[p][2]
        σ_y = -ρ⁰ * g * (H-y)
        σ_x = σ_y * ν / (1-ν)
        particles.σ[p] = diagm(Vec(σ_x, σ_y, σ_x))
    end
    @. particles.m = ρ⁰ * particles.V
    @. particles.F = one(particles.F)
    @show length(particles)

    ## Basis weights
    weights = generate_basis_weights(KernelCorrection(BSpline(Quadratic())), grid.x, length(particles))

    ## Material model
    model = DruckerPrager(; λ, G, ϕ, ψ)

    ## Outputs
    outdir = mkpath(joinpath("output", "collapse"))
    pvdfile = joinpath(outdir, "paraview")
    closepvd(openpvd(pvdfile)) # Create file

    t = 0.0
    step = 0
    fps = 60
    savepoints = collect(LinRange(t, t_stop, round(Int, t_stop*fps)+1))

    Tesserae.@showprogress while t < t_stop

        ## Calculate time step based on the wave speed
        vmax = maximum(@. sqrt((λ+2G) / (particles.m/particles.V)) + norm(particles.v))
        Δt = CFL * h / vmax

        ## Update basis weights
        update!(weights, particles, grid.x)

        ## Particle-to-grid transfer
        @P2G grid=>i particles=>p weights=>ip begin
            m[i]  = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
            f[i]  = @∑ -V[p] * resize(σ[p],(2,2)) * ∇w[ip] + w[ip] * m[p] * Vec(0,-g)
            m⁻¹[i] = inv(m[i]) * !iszero(m[i])
            vⁿ[i]  = mv[i] * m⁻¹[i]
            v[i]   = vⁿ[i] + (f[i] * m⁻¹[i]) * Δt
        end

        ## Boundary conditions
        for i in eachindex(grid)[:,begin]
            μ = 0.4 # Friction coefficient on the floor
            n = Vec(0,-1)
            vᵢ = grid.v[i]
            if !iszero(vᵢ)
                v̄ₙ = vᵢ ⋅ n
                vₜ = vᵢ - v̄ₙ*n
                v̄ₜ = norm(vₜ)
                grid.v[i] = vᵢ - (v̄ₙ*n + min(μ*v̄ₙ/v̄ₜ, 1) * vₜ)
            end
        end

        ## Grid-to-particle transfer
        @G2P grid=>i particles=>p weights=>ip begin
            v[p] += @∑ w[ip] * (v[i] - vⁿ[i])
            ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
            x[p] += @∑ w[ip] * v[i] * Δt
            ## Update Cauchy stress using Jaumann stress rate
            ∇uₚ = resize(∇v[p], (3,3)) * Δt
            σ[p] = cauchy_stress(model, σ[p], ∇uₚ)
            ## Update deformation gradient and volume
            ΔFₚ = I + ∇v[p] * Δt
            F[p] = ΔFₚ * F[p]
            V[p] = det(ΔFₚ) * V[p]
        end

        t += Δt
        step += 1

        if t > first(savepoints)
            popfirst!(savepoints)
            openpvd(pvdfile; append=true) do pvd
                openvtm(string(pvdfile, step)) do vtm
                    openvtk(vtm, particles.x) do vtk
                        vtk["Velocity (m/s)"] = particles.v
                        vtk["ID"] = eachindex(particles.v)
                    end
                    openvtk(vtm, grid.x) do vtk
                        vtk["Velocity (m/s)"] = grid.v
                    end
                    pvd[t] = vtm
                end
            end
        end
    end
    sum(particles.x) / length(particles) #src
end

using Test                             #src
if @isdefined(RUN_TESTS) && RUN_TESTS  #src
    @test main()[2] ≈ 0.1184 rtol=0.02 #src
end                                    #src
