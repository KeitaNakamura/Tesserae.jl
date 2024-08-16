# # Jacobian-free Newton--Krylov method
#
# ```@raw html
# <img src="https://github.com/user-attachments/assets/139ea30f-d1d7-4876-bc78-d5e6a1feea3e" width="400"/>
# ```

using Tesserae

using IterativeSolvers: gmres!
using LinearMaps: LinearMap

struct FLIP end
struct TPIC end

function main(transfer = FLIP())

    ## Simulation parameters
    h  = 0.1   # Grid spacing
    T  = 3.0   # Time span
    Δt = 0.002 # Timestep
    if @isdefined(RUN_TESTS) && RUN_TESTS #src
        T = 0.2                           #src
    end                                   #src

    ## Material constants
    E  = 1e6                    # Young's modulus
    ν  = 0.3                    # Poisson's ratio
    λ  = (E*ν) / ((1+ν)*(1-2ν)) # Lame's first parameter
    μ  = E / 2(1 + ν)           # Shear modulus
    ρ⁰ = 1000.0                 # Initial density

    ## Newmark-beta integration
    β = 1/4
    γ = 1/2

    GridProp = @NamedTuple begin
        X   :: Vec{3, Float64}
        m   :: Float64
        m⁻¹ :: Float64
        v   :: Vec{3, Float64}
        vⁿ  :: Vec{3, Float64}
        mv  :: Vec{3, Float64}
        a   :: Vec{3, Float64}
        aⁿ  :: Vec{3, Float64}
        ma  :: Vec{3, Float64}
        u   :: Vec{3, Float64}
        f   :: Vec{3, Float64}
        δu  :: Vec{3, Float64}
    end
    ParticleProp = @NamedTuple begin
        x    :: Vec{3, Float64}
        m    :: Float64
        V⁰   :: Float64
        v    :: Vec{3, Float64}
        a    :: Vec{3, Float64}
        ∇v   :: SecondOrderTensor{3, Float64, 9}
        ∇a   :: SecondOrderTensor{3, Float64, 9}
        ∇u   :: SecondOrderTensor{3, Float64, 9}
        F    :: SecondOrderTensor{3, Float64, 9}
        ΔF⁻¹ :: SecondOrderTensor{3, Float64, 9}
        τ    :: SecondOrderTensor{3, Float64, 9}
        c    :: FourthOrderTensor{3, Float64, 81}
    end

    ## Background grid
    grid = generate_grid(GridProp, CartesianMesh(h, (0,2), (-0.75,0.75), (-0.75,0.75)))

    ## Particles
    beam = Tesserae.Box((0,2), (-0.25,0.25), (-0.25,0.25))
    particles = generate_particles(ParticleProp, grid.X; domain=beam, spacing=1/3, alg=GridSampling())
    particles.V⁰ .= volume(beam) / length(particles)
    @. particles.m = ρ⁰ * particles.V⁰
    @. particles.F = one(particles.F)
    @show length(particles)

    ## Interpolation
    ## Use the kernel correction to properly handle the boundary conditions
    mpvalues = generate_mpvalues(KernelCorrection(BSpline(Quadratic())), grid.X, length(particles))

    ## Neo-Hookean model
    function stored_energy(C)
        dim = size(C, 1)
        J = √det(C)
        μ/2*(tr(C)-dim) - μ*log(J) + λ/2*(log(J))^2
    end
    function kirchhoff_stress(F)
        S = 2 * gradient(stored_energy, F' ⋅ F)
        symmetric(F ⋅ S ⋅ F')
    end

    ## Outputs
    outdir = mkpath(joinpath("output", "implicit_jacobian_free"))
    pvdfile = joinpath(outdir, "paraview")
    closepvd(openpvd(pvdfile)) # Create file

    t = 0.0
    step = 0
    fps = 60
    savepoints = collect(LinRange(t, T, round(Int, T*fps)+1))

    Tesserae.@showprogress while t < T

        for p in eachindex(particles, mpvalues)
            update!(mpvalues[p], particles.x[p], grid.X)
        end

        if transfer isa FLIP
            @P2G grid=>i particles=>p mpvalues=>ip begin
                m[i]  = @∑ w[ip] * m[p]
                mv[i] = @∑ w[ip] * m[p] * v[p]
                ma[i] = @∑ w[ip] * m[p] * a[p]
            end
        elseif transfer isa TPIC
            @P2G grid=>i particles=>p mpvalues=>ip begin
                m[i]  = @∑ w[ip] * m[p]
                mv[i] = @∑ w[ip] * m[p] * (v[p] + ∇v[p] ⋅ (X[i] - x[p]))
                ma[i] = @∑ w[ip] * m[p] * (a[p] + ∇a[p] ⋅ (X[i] - x[p]))
            end
        end

        ## Compute the grid velocity and acceleration at t = tⁿ
        @. grid.m⁻¹ = inv(grid.m) * !iszero(grid.m)
        @. grid.vⁿ = grid.mv * grid.m⁻¹
        @. grid.aⁿ = grid.ma * grid.m⁻¹

        ## Create dofmask
        dofmask = trues(3, size(grid)...)
        for i in eachindex(grid)
            iszero(grid.m[i]) && (dofmask[:,i] .= false)
        end

        ## Update boundary conditions
        @. grid.u = zero(grid.u)
        for i in eachindex(grid)[1,:,:]
            dofmask[:,i] .= false
        end
        for i in eachindex(grid)[end,:,:]
            if t < 1.0
                dofmask[:,i] .= false
                grid.u[i] = rotate(grid.X[i], rotmat(2π*Δt, Vec(1,0,0))) - grid.X[i]
            else
                dofmask[1,i] = false
            end
        end
        dofmap = DofMap(dofmask)

        ## Solve the nonlinear equation
        state = (; grid, particles, mpvalues, kirchhoff_stress, β, γ, dofmap, Δt)
        U = copy(dofmap(grid.u)) # Convert grid data to plain vector data
        compute_residual(U) = residual(U, state)
        compute_jacobian(U) = jacobian(U, state)
        Tesserae.newton!(U, compute_residual, compute_jacobian; linsolve = (x,A,b)->gmres!(x,A,b))

        ## Grid dispacement, velocity and acceleration have been updated during Newton's iterations
        if transfer isa FLIP
            @G2P grid=>i particles=>p mpvalues=>ip begin
                v[p] += @∑ Δt * w[ip] * ((1-γ)*a[p] + γ*a[i])
                a[p]  = @∑ w[ip] * a[i]
                x[p]  = @∑ w[ip] * (X[i] + u[i])
                ∇u[p] = @∑ u[i] ⊗ ∇w[ip]
                F[p]  = (I + ∇u[p]) ⋅ F[p]
            end
        elseif transfer isa TPIC
            @G2P grid=>i particles=>p mpvalues=>ip begin
                v[p]  = @∑ w[ip] * v[i]
                a[p]  = @∑ w[ip] * a[i]
                x[p]  = @∑ w[ip] * (X[i] + u[i])
                ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
                ∇a[p] = @∑ a[i] ⊗ ∇w[ip]
                ∇u[p] = @∑ u[i] ⊗ ∇w[ip]
                F[p]  = (I + ∇u[p]) ⋅ F[p]
            end
        end

        t += Δt
        step += 1

        if t > first(savepoints)
            popfirst!(savepoints)
            openpvd(pvdfile; append=true) do pvd
                openvtk(string(pvdfile, step), particles.x) do vtk
                    vtk["Velocity (m/s)"] = particles.v
                    vtk["von Mises stress (kPa)"] = @. 1e-3 * vonmises(particles.τ / det(particles.F))
                    pvd[t] = vtk
                end
            end
        end
    end
    Wₖ = sum(pt -> pt.m * (pt.v ⋅ pt.v) / 2, particles)                        #src
    Wₑ = sum(pt -> pt.V⁰ * det(pt.F) * stored_energy(pt.F' ⋅ pt.F), particles) #src
    Wₖ + Wₑ                                                                    #src
end

function residual(U::AbstractVector, state)
    (; grid, particles, mpvalues, kirchhoff_stress, β, γ, dofmap, Δt) = state

    dofmap(grid.u) .= U
    @. grid.a = (1/(β*Δt^2))*grid.u - (1/(β*Δt))*grid.vⁿ - (1/2β-1)*grid.aⁿ
    @. grid.v = grid.vⁿ + Δt*((1-γ)*grid.aⁿ + γ*grid.a)

    transposing_tensor(σ) = @einsum (i,j,k,l) -> σ[i,l] * one(σ)[j,k]
    @G2P grid=>i particles=>p mpvalues=>ip begin
        ## In addition to updating the stress tensor, the stiffness tensor,
        ## which is utilized in the Jacobian-vector product, is also updated.
        ∇u[p] = @∑ u[i] ⊗ ∇w[ip]
        ΔF⁻¹[p] = inv(I + ∇u[p])
        c[p], τ[p] = gradient(∇u -> kirchhoff_stress((I + ∇u) ⋅ F[p]), ∇u[p], :all)
        c[p] = c[p] - transposing_tensor(τ[p] ⋅ ΔF⁻¹[p]')
    end
    @P2G grid=>i particles=>p mpvalues=>ip begin
        f[i] = @∑ -V⁰[p] * τ[p] ⋅ (∇w[ip] ⋅ ΔF⁻¹[p])
    end

    @. β*Δt^2 * ($dofmap(grid.a) - $dofmap(grid.f) * $dofmap(grid.m⁻¹))
end

function jacobian(U::AbstractVector, state)
    (; grid, particles, mpvalues, β, dofmap, Δt) = state

    ## Create a linear map to represent Jacobian-vector product J*δU.
    ## `U` is acutally not used because the stiffness tensor is already calculated
    ## when computing the residual vector.
    fillzero!(grid.δu)
    LinearMap(ndofs(dofmap)) do JδU, δU
        dofmap(grid.δu) .= δU

        @G2P grid=>i particles=>p mpvalues=>ip begin
            ∇u[p] = @∑ δu[i] ⊗ ∇w[ip]
            τ[p] = c[p] ⊡ ∇u[p]
        end
        @P2G grid=>i particles=>p mpvalues=>ip begin
            f[i] = @∑ -V⁰[p] * τ[p] ⋅ (∇w[ip] ⋅ ΔF⁻¹[p])
        end

        @. JδU = δU - β*Δt^2 * $dofmap(grid.f) * $dofmap(grid.m⁻¹)
    end
end

using Test                                #src
if @isdefined(RUN_TESTS) && RUN_TESTS     #src
    @test main(FLIP()) ≈ 1635.9 rtol=1e-4 #src
    @test main(TPIC()) ≈ 1487.1 rtol=1e-4 #src
end                                       #src
