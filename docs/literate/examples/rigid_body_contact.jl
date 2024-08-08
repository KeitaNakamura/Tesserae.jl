# # Frictional contact with rigid body

using Tesserae

mutable struct Disk{dim, T}
    x::Vec{dim, T}
    v::Vec{dim, T}
end

function main()

    ## Simulation parameters
    h   = 0.004 # Grid spacing
    T   = 5.0   # Time span
    g   = 9.81  # Gravity acceleration
    CFL = 1.0   # Courant number
    if @isdefined(RUN_TESTS) && RUN_TESTS #src
        h = 0.008                         #src
        T = 2.0                           #src
    end                                   #src

    ## Material constants
    E  = 1000e3                 # Young's modulus
    ν  = 0.49                   # Poisson's ratio
    λ  = (E*ν) / ((1+ν)*(1-2ν)) # Lame's first parameter
    G  = E / 2(1 + ν)           # Shear modulus
    σy = 1.0e3                  # Yield stress
    ρ⁰ = 1.0e3                  # Initial density

    ## Disk
    D = 0.04
    disk = Disk(Vec(0, 7.5D), Vec(0, -0.25D))

    ## Contact parameters
    k = 1e6 # Penalty coefficient
    μ = 0.6 # Friction coefficient

    ## Utils
    @inline function contact_force_normal(x, r, x_disk)
        d = x - x_disk
        k * max(D/2 - (norm(d)-r), 0) * normalize(d)
    end
    @inline function contact_force_tangential(fₙ, v, m, Δt)
        iszero(fₙ) && return zero(fₙ)
        n = normalize(fₙ)
        fₜ = -m * (v-(v⋅n)*n) / Δt # Sticking force
        min(1, μ*norm(fₙ)/norm(fₜ)) * fₜ
    end

    ## Properties for grid and particles
    GridProp = @NamedTuple begin
        x    :: Vec{2, Float64}
        m    :: Float64
        m⁻¹  :: Float64
        mv   :: Vec{2, Float64}
        fint :: Vec{2, Float64}
        fext :: Vec{2, Float64}
        v    :: Vec{2, Float64}
        vⁿ   :: Vec{2, Float64}
    end
    ParticleProp = @NamedTuple begin
        x  :: Vec{2, Float64}
        m  :: Float64
        V  :: Float64
        r  :: Float64
        v  :: Vec{2, Float64}
        ∇v :: SecondOrderTensor{2, Float64, 4}
        F  :: SecondOrderTensor{3, Float64, 9}
        σ  :: SymmetricSecondOrderTensor{3, Float64, 6}
        ϵ  :: SymmetricSecondOrderTensor{3, Float64, 6}
        b  :: Vec{2, Float64}
    end

    ## Background grid
    H = 7D   # Ground height
    grid = generate_grid(GridProp, CartesianMesh(h, (0,5D), (0,H+D)))

    ## Particles
    particles = generate_particles(ParticleProp, grid.x)
    disk_points = filter(particles.x) do (x,y) # Points representing a disk just for visualization
        x^2 + (y-7.5D)^2 < (D/2)^2
    end
    particles.V .= volume(grid.x) / length(particles)
    filter!(pt -> pt.x[2] < H, particles)
    @. particles.m = ρ⁰ * particles.V
    @. particles.r = particles.V^(1/2) / 2
    @. particles.b = Vec(0,-g)
    @. particles.F = one(particles.F)
    for p in eachindex(particles)
        x, y = particles.x[p]
        σ_y = -ρ⁰ * g * (H - y)
        σ_x = ν/(1-ν) * σ_y
        particles.σ[p] = [σ_x 0.0 0.0
                          0.0 σ_y 0.0
                          0.0 0.0 σ_x]
    end
    @show length(particles)

    ## Interpolation
    mpvalues = generate_mpvalues(KernelCorrection(BSpline(Quadratic())), grid.x, length(particles))

    ## Output
    outdir = mkpath(joinpath("output", "rigid_body_contact"))
    pvdfile = joinpath(outdir, "paraview")
    closepvd(openpvd(pvdfile)) # Create file

    t = 0.0
    step = 0
    fps = 20
    savepoints = collect(LinRange(t, T, round(Int, T*fps)+1))

    Tesserae.@showprogress while t < T

        vmax = maximum(@. sqrt((λ+2G) / (particles.m/particles.V)) + norm(particles.v))
        Δt = CFL * spacing(grid) / vmax

        for p in eachindex(particles, mpvalues)
            update!(mpvalues[p], particles.x[p], grid.x)
        end

        @P2G grid=>i particles=>p mpvalues=>ip begin
            m[i]  = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * (v[p] + ∇v[p] ⋅ (x[i] - x[p])) # Taylor transfer
            fint[i] = @∑ -V[p] * resizedim(σ[p], Val(2)) ⋅ ∇w[ip] + w[ip] * m[p] * b[p]
            fext[i] = @∑ w[ip] * contact_force_normal(x[p], r[p], disk.x)
        end

        @. grid.m⁻¹ = inv(grid.m) * !iszero(grid.m)
        @. grid.vⁿ  = grid.mv * grid.m⁻¹
        @. grid.v   = grid.vⁿ + Δt * grid.fint * grid.m⁻¹
        @. grid.fext += contact_force_tangential(grid.fext, grid.v-disk.v, grid.m, Δt)
        @. grid.v    += Δt * grid.fext * grid.m⁻¹

        for i in eachindex(grid)[[begin,end],:]
            grid.v[i] = grid.v[i] .* (false,true)
        end
        for i in eachindex(grid)[:,[begin,end]]
            grid.v[i] = grid.v[i] .* (false,false)
        end

        @G2P grid=>i particles=>p mpvalues=>ip begin
            v[p]  = @∑ w[ip] * v[i] # PIC transfer
            ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
            x[p] += Δt * v[p]
        end

        for p in 1:length(particles)
            Δϵ = resizedim(Δt * symmetric(particles.∇v[p]), Val(3))
            particles.σ[p]  = vonmises_model(particles.σ[p], Δϵ; λ, G, σy)
            particles.V[p] *= 1 + tr(Δϵ)
            particles.ϵ[p] += Δϵ
        end

        disk.x += disk.v * Δt
        @. disk_points += disk.v * Δt

        t += Δt
        step += 1

        if t > first(savepoints)
            popfirst!(savepoints)
            openpvd(pvdfile; append=true) do pvd
                openvtm(string(pvdfile, step)) do vtm
                    deviatoric_strain(ϵ) = sqrt(2/3 * dev(ϵ) ⊡ dev(ϵ))
                    openvtk(vtm, particles.x) do vtk
                        vtk["vonmises"] = @. vonmises(particles.σ)
                        vtk["deviatoric strain"] = @. deviatoric_strain(particles.ϵ)
                    end
                    openvtk(vtm, disk_points) do vtk
                    end
                    pvd[t] = vtm
                end
            end
        end
    end
    sum(grid.fext) #src
end

function vonmises_model(σⁿ, Δϵ; λ, G, σy)
    δ = one(SymmetricSecondOrderTensor{3})
    I = one(SymmetricFourthOrderTensor{3})
    cᵉ = λ*δ⊗δ + 2G*I
    σ_trial = σⁿ + cᵉ ⊡ Δϵ
    dfdσ, f_trial = gradient(σ -> vonmises(σ) - σy, σ_trial, :all)
    if f_trial > 0
        dλ = f_trial / (dfdσ ⊡ cᵉ ⊡ dfdσ)
        σ = σ_trial - cᵉ ⊡ (dλ * dfdσ)
    else
        σ = σ_trial
    end
    if mean(σ) > 0 # simple tension cut-off
        σ = dev(σ)
    end
    σ
end

using Test                            #src
if @isdefined(RUN_TESTS) && RUN_TESTS #src
    @test main() ≈ [28,-92] rtol=0.2  #src
end                                   #src
