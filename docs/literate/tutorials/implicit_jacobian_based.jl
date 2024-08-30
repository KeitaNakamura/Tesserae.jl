# # Jacobian-based implicit method
#
# ```@raw html
# <img src="https://github.com/user-attachments/assets/44eaa91f-ff22-4768-abe5-9fd76079612a" width="300"/>
# ```
#
# | # Particles | # Iterations | Execution time |
# | ----------- | ------------ | -------------- |
# | 600         | 300          | 3 sec          |

using Tesserae
using LinearAlgebra

function main()

    ## Simulation parameters
    h  = 0.25 # Grid spacing
    T  = 3.0  # Time span
    g  = 10.0 # Gravity acceleration
    Δt = 0.01 # Time step

    ## Material constants
    E  = 1e6                    # Young's modulus
    ν  = 0.3                    # Poisson's ratio
    λ  = (E*ν) / ((1+ν)*(1-2ν)) # Lame's first parameter
    μ  = E / 2(1 + ν)           # Shear modulus
    ρ⁰ = 1050.0                 # Density

    ## Newmark-beta integration
    β = 1/4
    γ = 1/2

    GridProp = @NamedTuple begin
        X   :: Vec{2, Float64}
        m   :: Float64
        m⁻¹ :: Float64
        v   :: Vec{2, Float64}
        vⁿ  :: Vec{2, Float64}
        mv  :: Vec{2, Float64}
        a   :: Vec{2, Float64}
        aⁿ  :: Vec{2, Float64}
        ma  :: Vec{2, Float64}
        u   :: Vec{2, Float64}
        f   :: Vec{2, Float64}
    end
    ParticleProp = @NamedTuple begin
        x    :: Vec{2, Float64}
        m    :: Float64
        V⁰   :: Float64
        v    :: Vec{2, Float64}
        a    :: Vec{2, Float64}
        b    :: Vec{2, Float64}
        ∇u   :: SecondOrderTensor{2, Float64, 4}
        F    :: SecondOrderTensor{2, Float64, 4}
        ΔF⁻¹ :: SecondOrderTensor{2, Float64, 4}
        τ    :: SecondOrderTensor{2, Float64, 4}
        ℂ    :: FourthOrderTensor{2, Float64, 16}
    end

    ## Background grid
    grid = generate_grid(GridProp, CartesianMesh(h, (-1,5), (-5,1)))

    ## Particles
    beam = extract(grid.X, (0,4), (0,1))
    particles = generate_particles(ParticleProp, beam; alg=GridSampling(spacing=1/3))
    particles.V⁰ .= volume(beam) / length(particles)
    @. particles.m = ρ⁰ * particles.V⁰
    @. particles.F = one(particles.F)
    @. particles.b = Vec(0,-g)
    @show length(particles)
    η = 1/3                                                                       #src
    corner_p = argmin(p -> norm(particles.x[p] - Vec(4,0)), eachindex(particles)) #src
    corner_0 = particles.x[corner_p] + η*h/2*Vec(1,-1)                            #src

    ## Interpolation
    ## Use the kernel correction to properly handle the boundary conditions
    it = KernelCorrection(BSpline(Quadratic()))
    mpvalues = generate_mpvalues(it, grid.X, length(particles))

    ## Neo-Hookean model
    function kirchhoff_stress(F)
        J = det(F)
        b = symmetric(F ⋅ F')
        μ*(b-I) + λ*log(J)*I
    end

    ## Sparse matrix
    A = create_sparse_matrix(it, grid.X)

    ## Outputs
    outdir = mkpath(joinpath("output", "implicit_jacobian_based"))
    pvdfile = joinpath(outdir, "paraview")
    closepvd(openpvd(pvdfile)) # Create file

    t = 0.0
    step = 0
    fps = 60
    savepoints = collect(LinRange(t, T, round(Int, T*fps)+1))

    Tesserae.@showprogress while t < T

        activenodes = trues(size(grid))
        for i in eachindex(grid)
            if grid.X[i][1] < 0 && grid.X[i][2] > -(1+2h)
                activenodes[i] = false
            end
        end
        for p in eachindex(particles, mpvalues)
            update!(mpvalues[p], particles.x[p], grid.X, activenodes)
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
        dofmask = trues(2, size(grid)...)
        for i in eachindex(grid)
            iszero(grid.m[i]) && (dofmask[:,i] .= false)
        end

        ## Update boundary conditions and dofmap
        @. grid.u = zero(grid.u)
        for i in eachindex(grid)
            if grid.X[i][1] ≤ 0 && grid.X[i][2] > -(1+2h)
                dofmask[:,i] .= false
            end
        end
        dofmap = DofMap(dofmask)

        ## Solve the nonlinear equation
        state = (; grid, particles, mpvalues, kirchhoff_stress, β, γ, A, dofmap, Δt)
        U = copy(dofmap(grid.u)) # Convert grid data to plain vector data
        compute_residual(U) = residual(U, state)
        compute_jacobian(U) = jacobian(U, state)
        Tesserae.newton!(U, compute_residual, compute_jacobian)

        ## Grid dispacement, velocity and acceleration have been updated during Newton's iterations
        @G2P grid=>i particles=>p mpvalues=>ip begin
            ∇u[p] = @∑ u[i] ⊗ ∇w[ip]
            a[p]  = @∑ w[ip] * a[i]
            v[p] += @∑ w[ip] * ((1-γ)*a[p] + γ*a[i]) * Δt
            x[p]  = @∑ w[ip] * (X[i] + u[i])
            F[p]  = (I + ∇u[p]) ⋅ F[p]
        end

        t += Δt
        step += 1

        if t > first(savepoints)
            popfirst!(savepoints)
            openpvd(pvdfile; append=true) do pvd
                openvtk(string(pvdfile, step), particles.x) do vtk
                    function stress3x3(F)
                        z = zero(Mat{2,1})
                        F3x3 = [F  z
                                z' 1]
                        kirchhoff_stress(F3x3) * inv(det(F3x3))
                    end
                    vtk["Velocity (m/s)"] = particles.v
                    vtk["von Mises stress (kPa)"] = @. 1e-3 * vonmises(stress3x3(particles.F))
                    pvd[t] = vtk
                end
            end
        end
    end
    lx = particles.F[corner_p] ⋅ Vec(η*h,0)                 #src
    ly = particles.F[corner_p] ⋅ Vec(0,η*h)                 #src
    disp = particles.x[corner_p] + (lx/2 - ly/2) - corner_0 #src
    disp[2]                                                 #src
end

function residual(U::AbstractVector, state)
    (; grid, particles, mpvalues, kirchhoff_stress, β, γ, dofmap, Δt) = state

    dofmap(grid.u) .= U
    @. grid.a = (1/(β*Δt^2))*grid.u - (1/(β*Δt))*grid.vⁿ - (1/2β-1)*grid.aⁿ
    @. grid.v = grid.vⁿ + ((1-γ)*grid.aⁿ + γ*grid.a) * Δt

    geometric(τ) = @einsum (i,j,k,l) -> τ[i,l] * one(τ)[j,k]
    @G2P grid=>i particles=>p mpvalues=>ip begin
        ## In addition to updating the stress tensor, the stiffness tensor,
        ## which is utilized in the Jacobian-vector product, is also updated.
        ∇u[p] = @∑ u[i] ⊗ ∇w[ip]
        ΔF⁻¹[p] = inv(I + ∇u[p])
        F = (I + ∇u[p]) ⋅ F[p]
        ∂τ∂F, τ = gradient(kirchhoff_stress, F, :all)
        τ[p] = τ
        ℂ[p] = ∂τ∂F ⋅ F' - geometric(τ)
    end
    @P2G grid=>i particles=>p mpvalues=>ip begin
        f[i] = @∑ V⁰[p] * τ[p] ⋅ (∇w[ip] ⋅ ΔF⁻¹[p]) - w[ip] * m[p] * b[p]
    end

    @. $dofmap(grid.m) * $dofmap(grid.a) + $dofmap(grid.f)
end

function jacobian(U::AbstractVector, state)
    (; grid, particles, mpvalues, β, A, dofmap, Δt) = state

    dotdot(a,ℂ,b) = @einsum (i,j) -> a[k] * ℂ[i,k,j,l] * b[l]
    @P2G_Matrix grid=>(i,j) particles=>p mpvalues=>(ip,jp) begin
        A[i,j] = @∑ dotdot(∇w[ip] ⋅ ΔF⁻¹[p], ℂ[p], ∇w[jp] ⋅ ΔF⁻¹[p]) * V⁰[p]
    end

    extract(A, dofmap) + Diagonal(inv(β*Δt^2) * dofmap(grid.m))
end

using Test                            #src
if @isdefined(RUN_TESTS) && RUN_TESTS #src
    @test main() ≈ -0.8325 rtol=1e-4  #src
end                                   #src
