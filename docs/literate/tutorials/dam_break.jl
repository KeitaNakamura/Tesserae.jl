# # Stabilized mixed MPM for incompressible fluid flow
#
# ```@raw html
# <img src="https://github.com/user-attachments/assets/76fd800e-fda7-4d89-afcd-9a8a2178ab41" width="600"/>
# ```
#
# | # Particles | # Iterations | Execution time |
# | ----------- | ------------ | -------------- |
# | 16k         | 3.5k         | 7 min          |
#
# This example employs stabilized mixed MPM with the variational multiscale method[^1].
#
# [^1]: [Chandra, B., Hashimoto, R., Matsumi, S., Kamrin, K. and Soga, K., 2024. Stabilized mixed material point method for incompressible fluid flow analysis. Computer Methods in Applied Mechanics and Engineering, 419, p.116644.](https://doi.org/10.1016/j.cma.2023.116644)

# ## Main function

using Tesserae
using LinearAlgebra

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
    cellcenter(cell, mesh) = mean(mesh[cellnodes(cell)])

    ## Properties for grid and particles
    GridProp = @NamedTuple begin
        X   :: Vec{2, Float64}
        x   :: Vec{2, Float64}
        m   :: Float64
        m⁻¹ :: Float64
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
    particles = generate_particles(ParticleProp, grid.X; alg=PoissonDiskSampling(spacing=1/3))
    particles.V .= volume(grid.X) / length(particles)
    filter!(pt -> pt.x[1]<1.2 && pt.x[2]<0.6, particles)
    @. particles.m = ρ * particles.V
    @. particles.b = Vec(0,-g)
    @show length(particles)

    ## Interpolation
    it = KernelCorrection(BSpline(Quadratic()))
    mpvalues = map(p -> MPValue(it, grid.X), eachindex(particles))
    mpvalues_cell = map(CartesianIndices(size(grid).-1)) do cell
        mp = MPValue(it, grid.X)
        xc = cellcenter(cell, grid.X)
        update!(mp, xc, grid.X)
    end

    ## Sparse matrix
    A = create_sparse_matrix(it, grid.X; ndofs=3)

    ## Output
    outdir = mkpath(joinpath("output", "dam_break"))
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
        for p in eachindex(mpvalues, particles)
            update!(mpvalues[p], particles.x[p], grid.X, activenodes)
        end
        for cell in CartesianIndices(size(grid) .- 1)
            xc = cellcenter(cell, grid.X)
            update!(mpvalues_cell[cell], xc, grid.X, activenodes)
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

        ## Truncate the negative mass caused by the kernel correction to zero.
        @. grid.m = max(grid.m, 0)

        @. grid.m⁻¹ = inv(grid.m) * !iszero(grid.m)
        @. grid.vⁿ = grid.mv * grid.m⁻¹
        @. grid.aⁿ = grid.ma * grid.m⁻¹

        ## Update a dof map
        dofmask = trues(3, size(grid)...)
        for i in eachindex(grid)
            dofmask[:,i] .= !iszero(grid.m[i])
        end
        for i in @view eachindex(grid)[[begin,end],:] # Walls
            grid.vⁿ[i] = grid.vⁿ[i] .* (false,true)
            grid.aⁿ[i] = grid.aⁿ[i] .* (false,true)
            dofmask[1,i] = false
        end
        for i in @view eachindex(grid)[:,begin] # Floor
            grid.vⁿ[i] = grid.vⁿ[i] .* (true,false)
            grid.aⁿ[i] = grid.aⁿ[i] .* (true,false)
            dofmask[2,i] = false
        end
        dofmap = DofMap(dofmask)

        ## Solve grid position, dispacement, velocity, acceleration and pressure by VMS method
        state = (; grid, particles, mpvalues, mpvalues_cell, ρ, μ, β, γ, A, dofmap, Δt)
        Δt′ = variational_multiscale_method(state)

        if transfer isa FLIP
            local α = transfer.α
            @G2P grid=>i particles=>p mpvalues=>ip begin
                v[p] = @∑ ((1-α)*v[i] + α*(v[p] + ((1-γ)*a[p] + γ*a[i])*Δt)) * w[ip]
                a[p] = @∑ a[i] * w[ip]
                x[p] = @∑ x[i] * w[ip]
            end
        elseif transfer isa TPIC
            @G2P grid=>i particles=>p mpvalues=>ip begin
                v[p] = @∑ v[i] * w[ip]
                a[p] = @∑ a[i] * w[ip]
                x[p] = @∑ x[i] * w[ip]
                ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
                ∇a[p] = @∑ a[i] ⊗ ∇w[ip]
            end
        end

        ## Particle shifting based on the δ-correction
        particle_shifting(state)

        ## Remove particles that accidentally move outside of the mesh
        outside = findall(x->!isinside(x, grid.X), particles.x)
        deleteat!(particles, outside)
        deleteat!(mpvalues, outside)

        t += Δt′
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
        end
    end
    # mean(particles.p) #src
    mean(particles.x) #src
end

# ## Variational multiscale method

function variational_multiscale_method(state)

    ## The simulation might fail occasionally due to regions with very small masses,
    ## such as splashes[^1]. Therefore, in this script, if Newton's method doesn't converge,
    ## a half time step is applied.

    (; grid, dofmap, Δt) = state
    @. grid.u_p = zero(grid.u_p)

    solved = false
    while !solved
        ## Reconstruct state using the current time step
        state = merge(state, (; Δt))

        ## Compute VMS stabilization coefficients using current grid velocity,
        ## which is used for Jacobian matrix
        grid.v .= grid.vⁿ
        compute_VMS_stabilization_coefficients(state)

        ## Try computing the Jacobian matrix and performing its LU decomposition.
        ## In this formulation[^1], the initial Jacobian matrix is used in all Newton's iterations.
        ## If the computation fails, use a smaller time step.
        J = lu(jacobian(state); check=false)
        issuccess(J) || (Δt /= 2; continue)
        
        ## Solve nonlinear system
        U = zeros(ndofs(dofmap)) # Initialize nodal dispacement and pressure with zero
        solved = Tesserae.newton!(U, U->residual(U,state), U->J;
                                  linsolve=(x,A,b)->ldiv!(x,A,b), atol=1e-10, rtol=1e-10)

        ## If the simulation fails to solve, retry with a smaller time step
        solved || (Δt /= 2)
    end

    ## Update the positions of grid nodes
    @. grid.x = grid.X + grid.u

    ## Return the acutally applied time step
    Δt
end

# ## VMS stabilization coefficients

function compute_VMS_stabilization_coefficients(state)
    (; grid, particles, mpvalues_cell, ρ, μ, Δt) = state

    c₁ = 4.0
    c₂ = 2.0
    τdyn = 1.0
    h = sqrt(4*spacing(grid.X)^2/π)

    ## In the following computation, `@G2P` is unavailable
    ## due to the use of `mpvalues_cell`
    for p in eachindex(particles)
        v̄ₚ = zero(eltype(particles.v))
        mp = mpvalues_cell[whichcell(particles.x[p], grid.X)]
        gridindices = neighboringnodes(mp, grid)
        for ip in eachindex(gridindices)
            i = gridindices[ip]
            v̄ₚ += mp.w[ip] * grid.v[i]
        end
        τ₁ = inv(ρ*τdyn/Δt + c₂*ρ*norm(v̄ₚ)/h + c₁*μ/h^2)
        τ₂ = h^2 / (c₁*τ₁)
        particles.τ₁[p] = τ₁
        particles.τ₂[p] = τ₂
    end
end

# ## δ-correction

function particle_shifting(state)
    (; grid, particles, mpvalues) = state

    @P2G grid=>i particles=>p mpvalues=>ip begin
        Ṽ[i] = @∑ V[p] * w[ip]
        E[i] = max(0, -V[i] + Ṽ[i])
    end

    E² = sum(E->E^2, grid.E)
    @G2P grid=>i particles=>p mpvalues=>ip begin
        ∇E²[p] = @∑ 2V[p] * E[i] * ∇w[ip]
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
    (; grid, particles, mpvalues, μ, β, γ, dofmap, Δt) = state

    ## Map `U` to grid dispacement and pressure
    dofmap(grid.u_p) .= U
    grid.u .= map(x->@Tensor(x[1:2]), grid.u_p)
    grid.p .= map(x->x[3], grid.u_p)

    ## Recompute nodal velocity and acceleration based on the Newmark-beta method
    @. grid.v = γ/(β*Δt)*grid.u - (γ/β-1)*grid.vⁿ - Δt/2*(γ/β-2)*grid.aⁿ
    @. grid.a = 1/(β*Δt^2)*grid.u - 1/(β*Δt)*grid.vⁿ - (1/2β-1)*grid.aⁿ

    ## Recompute particle properties for residual vector
    @G2P grid=>i particles=>p mpvalues=>ip begin
        a[p]  = @∑ a[i] * w[ip]
        p[p]  = @∑ p[i] * w[ip]
        ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
        ∇p[p] = @∑ p[i] * ∇w[ip]
        s[p]  = 2μ * symmetric(∇v[p])
    end

    ## Compute VMS stabilization coefficients based on the current nodal velocity
    compute_VMS_stabilization_coefficients(state)

    ## Compute residual values
    @P2G grid=>i particles=>p mpvalues=>ip begin
        R_mom[i]  = @∑ V[p]*s[p]⋅∇w[ip] - m[p]*b[p]*w[ip] - V[p]*p[p]*∇w[ip] + τ₂[p]*V[p]*tr(∇v[p])*∇w[ip]
        R_mas[i]  = @∑ V[p]*tr(∇v[p])*w[ip] + τ₁[p]*m[p]*(a[p]-b[p])⋅∇w[ip] + τ₁[p]*V[p]*∇p[p]⋅∇w[ip]
        R_mom[i] += m[i]*a[i]
    end

    ## Map grid values to vector `R`
    dofmap(map(vcat, grid.R_mom, grid.R_mas))
end

# ## Jacobian matrix

function jacobian(state)
    (; grid, particles, mpvalues, ρ, μ, β, γ, A, dofmap, Δt) = state

    ## Construct the Jacobian matrix
    cₚ = 2μ * one(SymmetricFourthOrderTensor{2})
    I(i,j) = ifelse(i===j, one(Mat{2,2}), zero(Mat{2,2}))
    @P2G_Matrix grid=>(i,j) particles=>p mpvalues=>(ip,jp) begin
        A[i,j] = @∑ begin
            Kᵤᵤ = (γ/(β*Δt) * ∇w[ip] ⋅ cₚ ⋅ ∇w[jp]) * V[p] + 1/(β*Δt^2) * I(i,j) * m[p] * w[jp]
            Kᵤₚ = -∇w[ip] * w[jp] * V[p]
            Kₚᵤ = (γ/(β*Δt)) * w[ip] * ∇w[jp] * V[p]
            K̂ᵤᵤ = γ/(β*Δt) * τ₂[p] * ∇w[ip] ⊗ ∇w[jp] * V[p]
            K̂ₚᵤ = 1/(β*Δt^2) * τ₁[p] * ρ * ∇w[ip] * w[jp] * V[p]
            K̂ₚₚ = τ₁[p] * ∇w[ip] ⋅ ∇w[jp] * V[p]
            [Kᵤᵤ+K̂ᵤᵤ    Kᵤₚ
             (Kₚᵤ+K̂ₚᵤ)' K̂ₚₚ]
        end
    end

    ## Extract the activated degrees of freedom
    extract(A, dofmap)
end

using Test                                            #src
if @isdefined(RUN_TESTS) && RUN_TESTS                 #src
    @test main(FLIP(1.0))  ≈ [0.645,0.259] rtol=0.005 #src
    @test main(TPIC())     ≈ [0.645,0.259] rtol=0.005 #src
end                                                   #src
