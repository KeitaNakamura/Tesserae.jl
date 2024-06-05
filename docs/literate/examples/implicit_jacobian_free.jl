# # Jacobian-free Newton--Krylov method

using Sequoia

using IterativeSolvers: gmres!
using LinearMaps: LinearMap

function main()

    ## Simulation parameters
    h  = 0.05 # Grid spacing
    T  = 1.0  # Time span
    g  = 20.0 # Gravity acceleration
    Δt = 0.02 # Timestep
    if @isdefined(RUN_TESTS) && RUN_TESTS #src
        h = 0.1                           #src
        T = 0.5                           #src
    end                                   #src

    ## Material constants
    E  = 1e6                    # Young's modulus
    ν  = 0.3                    # Poisson's ratio
    λ  = (E*ν) / ((1+ν)*(1-2ν)) # Lame's first parameter
    μ  = E / 2(1 + ν)           # Shear modulus
    ρ⁰ = 500.0                  # Density

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
    end
    ParticleProp = @NamedTuple begin
        x    :: Vec{3, Float64}
        m    :: Float64
        V⁰   :: Float64
        v    :: Vec{3, Float64}
        a    :: Vec{3, Float64}
        b    :: Vec{3, Float64}
        ∇u   :: SecondOrderTensor{3, Float64, 9}
        F    :: SecondOrderTensor{3, Float64, 9}
        ΔF⁻¹ :: SecondOrderTensor{3, Float64, 9}
        τ    :: SecondOrderTensor{3, Float64, 9}
        c    :: FourthOrderTensor{3, Float64, 81}
    end

    ## Background grid
    grid = generate_grid(GridProp, CartesianMesh(h, (0.0,1.2), (0.0,2.0), (-0.2,0.2)))

    ## Particles
    beam = Sequoia.Box((0,1), (0.85,1.15), (-0.15,0.15))
    particles = generate_particles(ParticleProp, grid.X; domain=beam)
    particles.V⁰ .= volume(beam) / length(particles)
    @. particles.m = ρ⁰ * particles.V⁰
    @. particles.F = one(particles.F)
    @. particles.b = Vec(0,-g,0)
    @show length(particles)

    ## Interpolation
    ## Use the kernel correction to properly handle the boundary conditions
    mpvalues = generate_mpvalues(Vec{3}, KernelCorrection(QuadraticBSpline()), length(particles))

    ## Neo-Hookean model
    function kirchhoff_stress(F)
        J = det(F)
        b = symmetric(F ⋅ F')
        μ*(b-I) + λ*log(J)*I
    end

    ## Outputs
    outdir = mkpath(joinpath("output", "implicit_jacobian_free"))
    pvdfile = joinpath(outdir, "paraview")
    closepvd(openpvd(pvdfile)) # Create file

    t = 0.0
    step = 0

    Sequoia.@showprogress while t < T

        for p in eachindex(particles, mpvalues)
            update!(mpvalues[p], particles.x[p], grid.X)
        end

        @P2G grid=>i particles=>p mpvalues=>ip begin
            m[i]  = @∑ N[ip] * m[p]
            mv[i] = @∑ N[ip] * m[p] * v[p]
            ma[i] = @∑ N[ip] * m[p] * a[p]
        end

        ## Compute the grid velocity and acceleration at t = tⁿ
        @. grid.m⁻¹ = inv(grid.m) * !iszero(grid.m)
        @. grid.vⁿ = grid.mv * grid.m⁻¹
        @. grid.aⁿ = grid.ma * grid.m⁻¹

        ## Update a dof map
        dofmask = trues(3, size(grid)...)
        for i in eachindex(grid)
            dofmask[:,i] .= !iszero(grid.m[i])
        end
        for i in eachindex(grid)[1,:,:]
            dofmask[:,i] .= false
            grid.vⁿ[i] = zero(Vec{3})
            grid.aⁿ[i] = zero(Vec{3})
        end
        dofmap = DofMap(dofmask)

        ## Solve the nonlinear equation
        state = (; grid, particles, mpvalues, kirchhoff_stress, β, γ, dofmap, Δt)
        @. grid.u = zero(grid.u) # Set zero dispacement for the first guess of the solution
        U = copy(dofmap(grid.u)) # Convert grid data to plain vector data
        compute_residual(U) = residual(U, state)
        compute_jacobian(U) = jacobian(U, state)
        Sequoia.newton!(U, compute_residual, compute_jacobian; linsolve = (x,A,b)->gmres!(x,A,b))

        ## Grid dispacement, velocity and acceleration have been updated during Newton's iterations
        @G2P grid=>i particles=>p mpvalues=>ip begin
            ∇u[p] = @∑ u[i] ⊗ ∇N[ip]
            a[p]  = @∑ a[i] * N[ip]
            v[p] += @∑ Δt*((1-γ)*a[p] + γ*a[i]) * N[ip]
            x[p]  = @∑ (X[i] + u[i]) * N[ip]
            F[p]  = (I + ∇u[p]) ⋅ F[p]
        end

        t += Δt
        step += 1

        openpvd(pvdfile; append=true) do pvd
            openvtk(string(pvdfile, step), particles.x) do vtk
                vtk["velocity"] = particles.v
                vtk["von Mises"] = @. vonmises(particles.τ / det(particles.F))
                pvd[t] = vtk
            end
        end
    end
    mean(particles.x) #src
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
        ∇u[p] = @∑ u[i] ⊗ ∇N[ip]
        ΔF⁻¹[p] = inv(I + ∇u[p])
        c[p], τ[p] = gradient(∇u -> kirchhoff_stress((I + ∇u) ⋅ F[p]), ∇u[p], :all)
        c[p] = c[p] - transposing_tensor(τ[p] ⋅ ΔF⁻¹[p]')
    end
    @P2G grid=>i particles=>p mpvalues=>ip begin
        f[i] = @∑ -V⁰[p] * τ[p] ⋅ (∇N[ip] ⋅ ΔF⁻¹[p]) + m[p] * b[p] * N[ip]
    end

    @. β*Δt^2 * ($dofmap(grid.a) - $dofmap(grid.f) * $dofmap(grid.m⁻¹))
end

function jacobian(U::AbstractVector, state)
    (; grid, particles, mpvalues, β, dofmap, Δt) = state

    ## Create a linear map to represent Jacobian-vector product J*δU.
    ## `U` is acutally not used because the stiffness tensor is already calculated
    ## when computing the residual vector.
    LinearMap(ndofs(dofmap)) do JδU, δU
        dofmap(grid.u) .= δU

        @G2P grid=>i particles=>p mpvalues=>ip begin
            ∇u[p] = @∑ u[i] ⊗ ∇N[ip]
            τ[p] = c[p] ⊡ ∇u[p]
        end
        @P2G grid=>i particles=>p mpvalues=>ip begin
            f[i] = @∑ -V⁰[p] * τ[p] ⋅ (∇N[ip] ⋅ ΔF⁻¹[p])
        end

        @. JδU = δU - β*Δt^2 * $dofmap(grid.f) * $dofmap(grid.m⁻¹)
    end
end

using Test                             #src
if @isdefined(RUN_TESTS) && RUN_TESTS  #src
    @test main() ≈ [0.5,1,0] rtol=0.02 #src
end                                    #src
