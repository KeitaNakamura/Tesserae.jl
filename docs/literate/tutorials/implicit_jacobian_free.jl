# # Jacobian-free Newton--Krylov method
#
# ```@raw html
# <img src="https://github.com/user-attachments/assets/f1d80c46-a8ff-44d4-ae82-768b480f25ea" width="800"/>
# ```
#
# | # Particles | # Iterations | Execution time |
# | ----------- | ------------ | -------------- |
# | 26k         | 300          | 2 min          |

using Tesserae

using IterativeSolvers: gmres!
using LinearMaps: LinearMap

function main()

    ## Simulation parameters
    h  = 0.05 # Grid spacing
    T  = 3.0  # Time span
    Δt = 0.01 # Timestep
    if @isdefined(RUN_TESTS) && RUN_TESTS #src
        h = 0.1                           #src
    end                                   #src

    ## Material constants
    E  = 100e3                  # Young's modulus
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
        ∇a   :: SecondOrderTensor{3, Float64, 9}
        ∇u   :: SecondOrderTensor{3, Float64, 9}
        F    :: SecondOrderTensor{3, Float64, 9}
        ΔF⁻¹ :: SecondOrderTensor{3, Float64, 9}
        τ    :: SecondOrderTensor{3, Float64, 9}
        c    :: FourthOrderTensor{3, Float64, 81}
    end

    ## Background grid
    grid = generate_grid(GridProp, CartesianMesh(h, (0,1.5), (-0.6,0.6), (-0.6,0.6)))

    ## Particles
    beam = extract(grid.X, (0,1.5), (-0.3,0.3), (-0.3,0.3))
    particles = generate_particles(ParticleProp, beam; spacing=1/6, alg=GridSampling())
    particles.V⁰ .= volume(beam) / length(particles)
    filter!(particles) do pt
        x, y, z = pt.x
        (-0.3<y<-0.25 || 0.25<y<0.3) && (-0.3<z<-0.25 || 0.25<z<0.3)
    end
    @. particles.m = ρ⁰ * particles.V⁰
    @. particles.F = one(particles.F)
    @show length(particles)

    ## Interpolation
    ## Use the kernel correction to properly handle the boundary conditions
    mpvalues = generate_mpvalues(KernelCorrection(BSpline(Quadratic())), grid.X, length(particles))

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
    fps = 60
    savepoints = collect(LinRange(t, T, round(Int, T*fps)+1))

    Tesserae.@showprogress while t < T

        for p in eachindex(particles, mpvalues)
            update!(mpvalues[p], particles.x[p], grid.X)
        end

        @P2G grid=>i particles=>p mpvalues=>ip begin
            m[i]  = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
            ma[i] = @∑ w[ip] * m[p] * a[p]
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
            dofmask[:,i] .= false
            grid.u[i] = (rotmat(2π*Δt, Vec(1,0,0)) - I) ⋅ grid.X[i]
        end
        dofmap = DofMap(dofmask)

        ## Solve the nonlinear equation
        state = (; grid, particles, mpvalues, kirchhoff_stress, β, γ, dofmap, Δt)
        U = copy(dofmap(grid.u)) # Convert grid data to plain vector data
        compute_residual(U) = residual(U, state)
        compute_jacobian(U) = jacobian(U, state)
        Tesserae.newton!(U, compute_residual, compute_jacobian; linsolve = (x,A,b)->gmres!(x,A,b))

        ## Grid dispacement, velocity and acceleration have been updated during Newton's iterations
        @G2P grid=>i particles=>p mpvalues=>ip begin
            v[p] += @∑ Δt * w[ip] * ((1-γ)*a[p] + γ*a[i])
            a[p]  = @∑ w[ip] * a[i]
            x[p]  = @∑ w[ip] * (X[i] + u[i])
            ∇u[p] = @∑ u[i] ⊗ ∇w[ip]
            F[p]  = (I + ∇u[p]) ⋅ F[p]
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
    mean(particles.x)
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
    @test main() ≈ [0.7406,0,0] rtol=1e-4 #src
end                                       #src
