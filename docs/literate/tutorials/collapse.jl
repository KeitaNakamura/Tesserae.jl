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
using CUDA

@kwdef struct DruckerPrager
    λ  :: Float32            # Lame's first parameter
    G  :: Float32            # Shear modulus
    ϕ  :: Float32            # Internal friction angle
    ψ  :: Float32 = ϕ        # Dilatancy angle
    c  :: Float32 = 0.0      # Cohesion
    pₜ :: Float32 = c/tan(ϕ) # Mean stress for tension limit
    ## Assume plane strain condition
    A  :: Float32 = 3√2c      / sqrt(9+12tan(ϕ)^2)
    B  :: Float32 = 3√2tan(ϕ) / sqrt(9+12tan(ϕ)^2)
    b  :: Float32 = 3√2tan(ψ) / sqrt(9+12tan(ψ)^2)
end

function cauchy_stress(model::DruckerPrager, σⁿ::SymmetricSecondOrderTensor{3}, ∇u::SecondOrderTensor{3})
    δ = one(SymmetricSecondOrderTensor{3, Float32})
    I = one(SymmetricFourthOrderTensor{3, Float32})

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
    h::Float32   = 0.001 # Grid spacing
    T::Float32   = 1.5  # Time span
    g::Float32   = 9.81 # Gravity acceleration
    CFL::Float32 = 0.5  # Courant number
    # if @isdefined(RUN_TESTS) && RUN_TESTS #src
        # h::Float32 = 0.015                         #src
    # end                                   #src

    ## Material constants
    E::Float32  = 1e6                    # Young's modulus
    ν::Float32  = 0.3                    # Poisson's ratio
    λ::Float32  = (E*ν) / ((1+ν)*(1-2ν)) # Lame's first parameter
    G::Float32  = E / 2(1 + ν)           # Shear modulus
    ϕ::Float32  = deg2rad(32)            # Internal friction angle
    ψ::Float32  = deg2rad(0)             # Dilatancy angle
    ρ⁰::Float32 = 1.5e3                  # Initial density

    ## Geometry
    H::Float32 = 0.8 # Height of sand column
    W::Float32 = 0.6 # Width of sand column

    GridProp = @NamedTuple begin
        x  :: Vec{2, Float32}
        m  :: Float32
        m⁻¹ :: Float32
        v  :: Vec{2, Float32}
        vⁿ :: Vec{2, Float32}
        mv :: Vec{2, Float32}
        f  :: Vec{2, Float32}
    end
    ParticleProp = @NamedTuple begin
        x  :: Vec{2, Float32}
        m  :: Float32
        V  :: Float32
        v  :: Vec{2, Float32}
        ∇v :: SecondOrderTensor{2, Float32, 4}
        F  :: SecondOrderTensor{2, Float32, 4}
        σ  :: SymmetricSecondOrderTensor{3, Float32, 6}
    end

    ## Background grid
    grid = generate_grid(GridProp, CartesianMesh(Float32, h, (-3,3), (0,1)))

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

    ## Interpolation weights
    weights = generate_interpolation_weights(Float32, KernelCorrection(BSpline(Quadratic())), grid.x, length(particles))

    ## Material model
    model = DruckerPrager(; λ, G, ϕ, ψ)

    ## Outputs
    outdir = mkpath(joinpath("output", "collapse"))
    pvdfile = joinpath(outdir, "paraview")
    closepvd(openpvd(pvdfile)) # Create file

    t::Float32 = 0.0
    step = 0
    fps = 60
    savepoints = collect(LinRange(t, T, round(Int, T*fps)+1))

    let (grid, particles, weights) = (grid, particles, weights) .|> gpu # move to GPU
        Tesserae.@showprogress while t < T

            ## Calculate time step based on the wave speed
            vmax = maximum(@. sqrt((λ+2G) / (particles.m/particles.V)) + norm(particles.v))
            Δt = CFL * h / vmax

            ## Update interpolation weights
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
            map!(grid.v, grid.x, grid.v) do xᵢ, vᵢ
                if xᵢ[2] == 0
                    μ::Float32 = 0.4 # Friction coefficient on the floor
                    n = Vec(0,-1)
                    if !iszero(vᵢ)
                        v̄ₙ = vᵢ ⋅ n
                        vₜ = vᵢ - v̄ₙ*n
                        v̄ₜ = norm(vₜ)
                        return vᵢ - (v̄ₙ*n + min(μ*v̄ₙ, v̄ₜ) * vₜ/v̄ₜ)
                    end
                end
                return vᵢ
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
                let (grid, particles) = (grid, particles) .|> cpu # move to CPU for output
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
        end
    end
    sum(particles.x) / length(particles) #src
end

using Test                             #src
if @isdefined(RUN_TESTS) && RUN_TESTS  #src
    @test main()[2] ≈ 0.1085 rtol=0.02 #src
end                                    #src
