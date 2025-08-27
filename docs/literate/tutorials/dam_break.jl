# # Stabilized mixed MPM for incompressible fluid flow
#
# ```@raw html
# <img src="https://github.com/user-attachments/assets/dfc9fb4e-6223-460e-ac34-310363cd6a78" width="600"/>
# ```
#
# | # Particles | # Iterations | Execution time (w/o output) |
# | ----------- | ------------ | --------------------------- |
# | 28k         | 3.5k         | 12 min                      |
#
# This example employs stabilized mixed MPM with the variational multiscale method[^1].
#
# [^1]: [Chandra, B., Hashimoto, R., Matsumi, S., Kamrin, K. and Soga, K., 2024. Stabilized mixed material point method for incompressible fluid flow analysis. Computer Methods in Applied Mechanics and Engineering, 419, p.116644.](https://doi.org/10.1016/j.cma.2023.116644)

# ## Main function

using Tesserae
using LinearAlgebra
using SparseArrays

using Krylov: gmres
using LinearOperators: LinearOperator
import AlgebraicMultigrid as AMG
using Graphs: SimpleGraph, connected_components

struct FLIP α::Float64 end
struct TPIC end

function main(transfer = FLIP(1.0))

    ## Simulation parameters
    h  = 0.02   # Grid spacing
    T  = 7.0    # Time span
    g  = 9.81   # Gravity acceleration
    Δt = 2.0e-3 # Time step
    if @isdefined(RUN_TESTS) && RUN_TESTS #src
        T = 0.2                           #src
    end                                   #src

    ## Material constants
    ρ = 1.0e3   # Initial density
    μ = 1.01e-3 # Dynamic viscosity (Pa⋅s)

    ## Newmark-beta method
    β = 0.5
    γ = 1.0

    ## Utils
    cellnodes(cell) = cell:(cell+oneunit(cell))
    cellcenter(cell, mesh) = sum(mesh[cellnodes(cell)]) / 4

    ## Properties for grid and particles
    GridProp = @NamedTuple begin
        X   :: Vec{2, Float64}
        x   :: Vec{2, Float64}
        m   :: Float64
        v   :: Vec{2, Float64}
        vⁿ  :: Vec{2, Float64}
        mv  :: Vec{2, Float64}
        a   :: Vec{2, Float64}
        aⁿ  :: Vec{2, Float64}
        ma  :: Vec{2, Float64}
        u   :: Vec{2, Float64}
        p   :: Float64
        ## δ-correction
        V   :: Float64
        Ṽ   :: Float64
        E   :: Float64
        ## Residuals
        u_p   :: Vec{3, Float64}
        R_mom :: Vec{2, Float64}
        R_mas :: Float64
    end
    ParticleProp = @NamedTuple begin
        x   :: Vec{2, Float64}
        m   :: Float64
        V   :: Float64
        v   :: Vec{2, Float64}
        ∇v  :: SecondOrderTensor{2, Float64, 4}
        a   :: Vec{2, Float64}
        ∇a  :: SecondOrderTensor{2, Float64, 4}
        p   :: Float64
        ∇p  :: Vec{2, Float64}
        s   :: SymmetricSecondOrderTensor{2, Float64, 3}
        b   :: Vec{2, Float64}
        ## δ-correction
        ∇E² :: Vec{2, Float64}
        ## Stabilization
        τ₁  :: Float64
        τ₂  :: Float64
    end

    ## Background grid
    grid = generate_grid(GridProp, CartesianMesh(h, (0,3.22), (0,2.5)))
    for cell in CartesianIndices(size(grid).-1)
        for i in cellnodes(cell)
            grid.V[i] += (h/2)^2
        end
    end

    ## Particles
    particles = generate_particles(ParticleProp, grid.X; alg=PoissonDiskSampling(spacing=1/4))
    particles.V .= volume(grid.X) / length(particles)
    filter!(pt -> pt.x[1]<1.2 && pt.x[2]<0.6, particles)
    @. particles.m = ρ * particles.V
    @. particles.b = Vec(0,-g)
    @show length(particles)

    ## Interpolation weights
    interp = KernelCorrection(BSpline(Quadratic()))
    weights = generate_interpolation_weights(interp, grid.X, length(particles); name=Val(:S))
    weights_cell = generate_interpolation_weights(interp, grid.X, size(grid) .- 1; name=Val(:S))

    ## Sparse matrix
    A = create_sparse_matrix(interp, grid.X; ndofs=3)

    partition = ColorPartition(grid.X)

    ## Output
    outdir = mkpath(joinpath(tempdir(), "dam_break"))
    pvdfile = joinpath(outdir, "paraview")
    closepvd(openpvd(pvdfile)) # Create file

    t = 0.0
    step = 0
    fps = 30
    savepoints = collect(LinRange(t, T, round(Int, T*fps)+1))

    Tesserae.@showprogress while t < T

        ## Update interpolation values based on the nodes of active cells
        ## where the particles are located
        activenodes = falses(size(grid))
        for p in eachindex(particles)
            cell = whichcell(particles.x[p], grid.X)
            activenodes[cellnodes(cell)] .= true
        end
        update!(weights, particles, grid.X, activenodes)
        for cell in CartesianIndices(size(grid) .- 1)
            xc = cellcenter(cell, grid.X)
            if all(i->activenodes[i], cellnodes(cell))
                update!(weights_cell[cell], xc, grid.X, activenodes)
            end
        end
        update!(partition, particles.x)

        if transfer isa FLIP
            @P2G grid=>i particles=>p weights=>ip begin
                m[i]  = @∑ S[ip] * m[p]
                mv[i] = @∑ S[ip] * m[p] * v[p]
                ma[i] = @∑ S[ip] * m[p] * a[p]
            end
        elseif transfer isa TPIC
            @P2G grid=>i particles=>p weights=>ip begin
                m[i]  = @∑ S[ip] * m[p]
                mv[i] = @∑ S[ip] * m[p] * (v[p] + ∇v[p] * (X[i] - x[p]))
                ma[i] = @∑ S[ip] * m[p] * (a[p] + ∇a[p] * (X[i] - x[p]))
            end
        end

        m_tol = sqrt(eps(eltype(grid.m))) * maximum(grid.m)
        m_mask = @. !(abs(grid.m) ≤ m_tol)
        @. grid.vⁿ = grid.mv / grid.m * m_mask
        @. grid.aⁿ = grid.ma / grid.m * m_mask

        ## Update a dof map
        dofmask_u = falses(3, size(grid)...)
        dofmask_p = falses(3, size(grid)...)
        for i in 1:2
            dofmask_u[i,:,:] = m_mask
        end
        dofmask_p[3,:,:] = m_mask
        for i in @view eachindex(grid)[[begin,end],:] # Walls
            grid.vⁿ[i] = grid.vⁿ[i] .* (false,true)
            grid.aⁿ[i] = grid.aⁿ[i] .* (false,true)
            dofmask_u[1,i] = false
        end
        for i in @view eachindex(grid)[:,begin] # Floor
            grid.vⁿ[i] = grid.vⁿ[i] .* (true,false)
            grid.aⁿ[i] = grid.aⁿ[i] .* (true,false)
            dofmask_u[2,i] = false
        end
        dofmap_u = DofMap(dofmask_u)
        dofmap_p = DofMap(dofmask_p)
        dofmap = DofMap(dofmask_u .| dofmask_p)

        ## Solve grid position, dispacement, velocity, acceleration and pressure by VMS method
        state = (; grid, particles, weights, weights_cell, ρ, μ, β, γ, A, dofmap, dofmap_u, dofmap_p, Δt, partition)
        variational_multiscale_method(state)

        if transfer isa FLIP
            α = transfer.α
            @G2P grid=>i particles=>p weights=>ip begin
                v[p] = @∑ ((1-α)*v[i] + α*(v[p] + ((1-γ)*a[p] + γ*a[i])*Δt)) * S[ip]
                a[p] = @∑ a[i] * S[ip]
                x[p] = @∑ x[i] * S[ip]
            end
        elseif transfer isa TPIC
            @G2P grid=>i particles=>p weights=>ip begin
                v[p] = @∑ v[i] * S[ip]
                a[p] = @∑ a[i] * S[ip]
                x[p] = @∑ x[i] * S[ip]
                ∇v[p] = @∑ v[i] ⊗ ∇S[ip]
                ∇a[p] = @∑ a[i] ⊗ ∇S[ip]
            end
        end

        ## Particle shifting based on the δ-correction
        particle_shifting(state)

        t += Δt
        step += 1

        ## Write results
        if t > first(savepoints)
            popfirst!(savepoints)
            openpvd(pvdfile; append=true) do pvd
                openvtm(string(pvdfile, step)) do vtm
                    vorticity(∇v) = ∇v[2,1] - ∇v[1,2]
                    openvtk(vtm, particles.x) do vtk
                        vtk["Pressure (Pa)"] = particles.p
                        vtk["Velocity (m/s)"] = particles.v
                        vtk["Vorticity (1/s)"] = vorticity.(particles.∇v)
                    end
                    openvtk(vtm, grid.X) do vtk
                    end
                    pvd[t] = vtm
                end
            end
            reorder_particles!(particles, partition)
        end

        ## Remove particles that accidentally move outside of the mesh
        outside = findall(x->!isinside(x, grid.X), particles.x)
        deleteat!(particles, outside)
    end
    # sum(particles.p) / length(particles) #src
    sum(particles.x) / length(particles) #src
end

# ## Variational multiscale method

function variational_multiscale_method(state)

    (; grid, dofmap, dofmap_u, dofmap_p, β, γ, Δt) = state
    @. grid.u_p = zero(grid.u_p)

    ## Compute VMS stabilization coefficients using current grid velocity,
    ## which is used for Jacobian matrix
    grid.v .= grid.vⁿ
    compute_VMS_stabilization_coefficients(state)

    ## Solve nonlinear system using GMRES with incomplete LU preconditioner
    K = jacobian(state)

    ## Extract the activated degrees of freedom
    A = extract(K, dofmap, dofmap)
    Aᵤᵤ = extract(K, dofmap_u, dofmap_u) # dispacement-dispacement
    Aᵤₚ = extract(K, dofmap_u, dofmap_p) # dispacement-pressure
    Aₚᵤ = extract(K, dofmap_p, dofmap_u) # pressure-dispacement
    Aₚₚ = extract(K, dofmap_p, dofmap_p) # pressure-pressure

    ## For multithreading
    Aᵀ = sparse(A')
    opA = LinearOperator(Float64, size(Aᵀ)..., false, false, (y, v) -> threaded_mul!(y, Aᵀ, v))

    ## Reindex DOFs relative to the extracted `A` (not the full `K`).
    dofs_u = indexin(dofs(dofmap_u), dofs(dofmap))
    dofs_p = indexin(dofs(dofmap_p), dofs(dofmap))

    ## Build block preconditioner (approximate Schur complement form)
    ## - Pᵤ: dispacement block preconditioner from Aᵤᵤ
    ## - Pₚ: pressure block preconditioner from approximate Schur complement
    xᵤ = zeros(size(Aᵤᵤ, 1))
    xₚ = zeros(size(Aₚₚ, 1))
    function coarse_solver(A)
        F = cholesky(Symmetric(A); check=false)
        !issuccess(F) && (F = qr(A))
        (x, b) -> copy!(x, F \ b)
    end
    δI = cbrt(eps(Float64)) * tr(Aₚₚ) / size(Aₚₚ, 1) * I
    S̃ = Aₚₚ - (-γ/(β*Δt)*Aᵤₚ') * inv(Diagonal(Aᵤᵤ)) * Aᵤₚ + δI
    comps = connected_components(SimpleGraph(Symmetric(S̃)))
    B = zeros(size(S̃,1), length(comps))
    for (j,c) in enumerate(comps)
        B[c,j] .= 1 / sqrt(length(c))
    end
    Pᵤ = AMG.aspreconditioner(AMG.smoothed_aggregation(Aᵤᵤ; presmoother=ThreadedJacobi(xᵤ), postsmoother=ThreadedJacobi(xᵤ), coarse_solver, max_coarse=10000))
    Pₚ = AMG.aspreconditioner(AMG.smoothed_aggregation(S̃; B, presmoother=ThreadedJacobi(xₚ), postsmoother=ThreadedJacobi(xₚ), coarse_solver, max_coarse=10000))

    ## Define operator of block preconditioner
    P⁻¹ = LinearOperator(Float64, size(A)..., false, false, (y, r) -> begin
        yᵤ = view(y, dofs_u); yₚ = view(y, dofs_p)
        rᵤ = view(r, dofs_u); rₚ = view(r, dofs_p)
        yᵤ .= Pᵤ \ Array(rᵤ)
        yₚ .= Pₚ \ (rₚ - (Aₚᵤ * Array(yᵤ)))
    end)

    U = zeros(ndofs(dofmap)) # Initialize nodal dispacement and pressure with zero
    linsolve(x, A, b) = copy!(x, gmres(A, b; N=P⁻¹, verbose=10)[1])
    Tesserae.newton!(U, U->residual(U,state), U->opA; linsolve, maxiter=20, verbose=true, backtracking=true)

    ## Update the positions of grid nodes
    @. grid.x = grid.X + grid.u
end

# ## VMS stabilization coefficients

function compute_VMS_stabilization_coefficients(state)
    (; grid, particles, weights_cell, ρ, μ, Δt) = state

    c₁ = 4.0
    c₂ = 2.0
    τdyn = 1.0
    h = sqrt(4*spacing(grid.X)^2/π)

    ## In the following computation, `@G2P` is unavailable
    ## due to the use of `weights_cell`
    for p in eachindex(particles)
        v̄ₚ = zero(eltype(particles.v))
        iw = weights_cell[whichcell(particles.x[p], grid.X)]
        gridindices = neighboringnodes(iw, grid)
        for ip in eachindex(gridindices)
            i = gridindices[ip]
            v̄ₚ += iw.S[ip] * grid.v[i]
        end
        τ₁ = inv(ρ*τdyn/Δt + c₂*ρ*norm(v̄ₚ)/h + c₁*μ/h^2)
        τ₂ = h^2 / (c₁*τ₁)
        particles.τ₁[p] = τ₁
        particles.τ₂[p] = τ₂
    end
end

# ## δ-correction

function particle_shifting(state)
    (; grid, particles, weights) = state

    @P2G grid=>i particles=>p weights=>ip begin
        Ṽ[i] = @∑ V[p] * S[ip]
        E[i] = max(0, -V[i] + Ṽ[i])
    end

    E² = sum(E->E^2, grid.E)
    @G2P grid=>i particles=>p weights=>ip begin
        ∇E²[p] = @∑ 2V[p] * E[i] * ∇S[ip]
    end

    b₀ = E² / sum(∇E²->∇E²⋅∇E², particles.∇E²)

    for p in eachindex(particles)
        xₚ = particles.x[p] - b₀ * particles.∇E²[p]
        if isinside(xₚ, grid.X)
            particles.x[p] = xₚ
        end
    end
end

# ## Residual vector

function residual(U, state)
    (; grid, particles, weights, μ, β, γ, dofmap, Δt, partition) = state

    ## Map `U` to grid dispacement and pressure
    dofmap(grid.u_p) .= U
    grid.u .= map(x->@Tensor(x[1:2]), grid.u_p)
    grid.p .= map(x->x[3], grid.u_p)

    ## Recompute nodal velocity and acceleration based on the Newmark-beta method
    @. grid.v = γ/(β*Δt)*grid.u - (γ/β-1)*grid.vⁿ - Δt/2*(γ/β-2)*grid.aⁿ
    @. grid.a = 1/(β*Δt^2)*grid.u - 1/(β*Δt)*grid.vⁿ - (1/2β-1)*grid.aⁿ

    ## Compute VMS stabilization coefficients based on the current nodal velocity
    compute_VMS_stabilization_coefficients(state)

    ## Compute residual values
    @threaded @G2P2G grid=>i particles=>p weights=>ip partition begin
        a[p]  = @∑ a[i] * S[ip]
        p[p]  = @∑ p[i] * S[ip]
        ∇v[p] = @∑ v[i] ⊗ ∇S[ip]
        ∇p[p] = @∑ p[i] * ∇S[ip]
        s[p]  = 2μ * symmetric(∇v[p])
        R_mom[i]  = @∑ V[p]*s[p]*∇S[ip] - m[p]*b[p]*S[ip] - V[p]*p[p]*∇S[ip] + τ₂[p]*V[p]*tr(∇v[p])*∇S[ip]
        R_mas[i]  = @∑ V[p]*tr(∇v[p])*S[ip] + τ₁[p]*m[p]*(a[p]-b[p])⋅∇S[ip] + τ₁[p]*V[p]*∇p[p]⋅∇S[ip]
        R_mom[i] += m[i]*a[i]
    end

    ## Map grid values to vector `R`
    Array(dofmap(map(vcat, grid.R_mom, grid.R_mas)))
end

# ## Jacobian matrix

function jacobian(state)
    (; grid, particles, weights, ρ, μ, β, γ, A, dofmap, Δt, partition) = state

    ## Construct the Jacobian matrix
    cₚ = 2μ * one(SymmetricFourthOrderTensor{2})
    I(i,j) = ifelse(i===j, one(Mat{2,2}), zero(Mat{2,2}))
    @threaded @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) partition begin
        A[i,j] = @∑ begin
            Kᵤᵤ = (γ/(β*Δt) * ∇S[ip] ⊡ cₚ ⊡ ∇S[jp]) * V[p] + 1/(β*Δt^2) * I(i,j) * m[p] * S[jp]
            Kᵤₚ = -∇S[ip] * S[jp] * V[p]
            Kₚᵤ = (γ/(β*Δt)) * S[ip] * ∇S[jp] * V[p]
            K̂ᵤᵤ = γ/(β*Δt) * τ₂[p] * ∇S[ip] ⊗ ∇S[jp] * V[p]
            K̂ₚᵤ = 1/(β*Δt^2) * τ₁[p] * ρ * ∇S[ip] * S[jp] * V[p]
            K̂ₚₚ = τ₁[p] * ∇S[ip] ⋅ ∇S[jp] * V[p]
            [Kᵤᵤ+K̂ᵤᵤ    Kᵤₚ
             (Kₚᵤ+K̂ₚᵤ)' K̂ₚₚ]
        end
    end

    A
end

function threaded_mul!(y::Vector{T}, A::SparseMatrixCSC{T}, x::Vector{T}) where T <: Number
    @threaded for i = 1 : A.n
        tmp = zero(T)
        @inbounds for j = A.colptr[i] : (A.colptr[i+1] - 1)
            tmp += A.nzval[j] * x[A.rowval[j]]
        end
        @inbounds y[i] = tmp
    end
    return y
end
function threaded_mul!(y::Vector{T}, A::Adjoint{T, <: SparseMatrixCSC{T}}, x::Vector{T}) where T <: Number
    threaded_mul!(y, parent(A), x)
end

struct ThreadedMatrix{T, M <: AbstractMatrix{T}} <: AbstractMatrix{T}
    data::M
end
Base.size(M::ThreadedMatrix) = size(M.data)
LinearAlgebra.mul!(Y::AbstractVector, A::ThreadedMatrix, B::AbstractVector) = threaded_mul!(Y, A.data, B)

function threaded_amg(ml)
    levels = [AMG.Level(ThreadedMatrix(l.A), l.P, ThreadedMatrix(l.R)) for l in ml.levels]
    AMG.MultiLevel(
        levels,
        ThreadedMatrix(ml.final_A),
        ml.coarse_solver,
        ml.presmoother,
        ml.postsmoother,
        ml.workspace,
    )
end

struct ThreadedJacobi{T,TX}
    ω::T
    temp::TX
    iter::Int
end
ThreadedJacobi(ω, x::TX; iter=1) where {T, TX<:AbstractArray{T}} = ThreadedJacobi{T,TX}(ω, similar(x), iter)
ThreadedJacobi(x::TX, ω=0.5; iter=1) where {T, TX<:AbstractArray{T}} = ThreadedJacobi{T,TX}(ω, similar(x), iter)

(jacobi::ThreadedJacobi)(A::ThreadedMatrix, x, b) = jacobi(A.data, x, b)
function (jacobi::ThreadedJacobi)(A, x, b)

    ω = jacobi.ω
    one = Base.one(eltype(A))
    temp = jacobi.temp
    z = zero(eltype(A))

    for k in 1:jacobi.iter
        for col = 1:size(x, 2)
            for i = 1:size(A, 1)
                @inbounds temp[i, col] = x[i, col]
            end

            @threaded for i = 1:size(A, 1)
                @inbounds begin
                    rsum = z
                    diag = z

                    for j in nzrange(A, i)
                        row = A.rowval[j]
                        val = A.nzval[j]

                        diag = ifelse(row == i, val, diag)
                        rsum += ifelse(row == i, z, val * temp[row, col])
                    end

                    xcand = (one - ω) * temp[i, col] + ω * ((b[i, col] - rsum) / diag)
                    x[i, col] = ifelse(diag == 0, x[i, col], xcand)
                end
            end
        end
    end
end

using Test                                            #src
if @isdefined(RUN_TESTS) && RUN_TESTS                 #src
    @test main(FLIP(1.0))  ≈ [0.645,0.259] rtol=0.005 #src
    @test main(TPIC())     ≈ [0.645,0.259] rtol=0.005 #src
end                                                   #src
