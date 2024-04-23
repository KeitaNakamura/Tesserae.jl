# # Stabilized mixed MPM for incompressible fluid

# ## Main function

using Sequoia
using LinearAlgebra

function main(transfer::Symbol = :FLIP)

    ## Simulation parameters
    h  = 0.02   # Grid spacing
    T  = 7.0    # Time span
    g  = 9.81   # Gravity acceleration
    Δt = 2.5e-3 # Timestep
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
            grid.V[i] += (spacing(grid)/2)^2
        end
    end

    ## Particles
    particles = generate_particles(ParticleProp, grid.X; spacing=1/3)
    particles.V .= volume(grid.X) / length(particles)
    filter!(pt -> pt.x[1]<1.2 && pt.x[2]<0.6, particles)
    @. particles.m = ρ * particles.V
    @. particles.b = Vec(0,-g)
    @show length(particles)

    ## Interpolation
    it = KernelCorrection(QuadraticBSpline())
    mpvalues = map(p -> MPValue(Vec{2}, it), eachindex(particles))
    mpvalues_cell = map(CartesianIndices(size(grid).-1)) do cell
        mp = MPValue(Vec{2}, it)
        xc = mean(grid.X[cellnodes(cell)])
        update!(mp, xc, grid.X)
    end

    ## BlockSpace for threaded computation
    blockspace = BlockSpace(grid.X)

    ## Sparse matrix
    A = create_sparse_matrix(Vec{3}, it, grid.X)

    ## Output
    outdir = mkpath(joinpath("output", "dam_break"))
    pvdfile = joinpath(outdir, "paraview")
    closepvd(openpvd(pvdfile)) # Create file

    t = 0.0
    step = 0
    fps = 30
    savepoints = collect(LinRange(t, T, round(Int, T*fps)+1))

    Sequoia.@showprogress while t < T

        ## Update interpolation values based on the active nodes
        ## of cells where the particles are located
        activenodes = falses(size(grid))
        for p in eachindex(particles)
            cell = whichcell(particles.x[p], grid.X)
            activenodes[cellnodes(cell)] .= true
        end
        for p in eachindex(mpvalues, particles)
            update!(mpvalues[p], particles.x[p], grid.X, activenodes)
        end

        update!(blockspace, particles.x)

        if transfer == :FLIP
            @P2G grid=>i particles=>p mpvalues=>ip begin
                m[i]  = @∑ N[ip] * m[p]
                mv[i] = @∑ N[ip] * m[p] * v[p]
                ma[i] = @∑ N[ip] * m[p] * a[p]
            end
        elseif transfer == :TPIC
            @P2G grid=>i particles=>p mpvalues=>ip begin
                m[i]  = @∑ N[ip] * m[p]
                mv[i] = @∑ N[ip] * m[p] * (v[p] + ∇v[p] ⋅ (X[i] - x[p]))
                ma[i] = @∑ N[ip] * m[p] * (a[p] + ∇a[p] ⋅ (X[i] - x[p]))
            end
        end

        ## Remove occasionally generated negative mass due to the kernel correction.
        ## This sufficiently reduces spurious pressure oscillations without sacrificing simulation accuracy.
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
        state = (; grid, particles, mpvalues, mpvalues_cell, blockspace, ρ, μ, β, γ, A, dofmap, Δt)
        Δt′ = variational_multiscale_method(state)

        if transfer == :FLIP
            @G2P grid=>i particles=>p mpvalues=>ip begin
                v[p] += @∑ (v[i] - vⁿ[i]) * N[ip]
                a[p]  = @∑ a[i] * N[ip]
                x[p]  = @∑ x[i] * N[ip]
            end
        elseif transfer == :TPIC
            @G2P grid=>i particles=>p mpvalues=>ip begin
                v[p]  = @∑ v[i] * N[ip]
                a[p]  = @∑ a[i] * N[ip]
                x[p]  = @∑ x[i] * N[ip]
                ∇v[p] = @∑ v[i] ⊗ ∇N[ip]
                ∇a[p] = @∑ a[i] ⊗ ∇N[ip]
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
                        vtk["pressure"] = particles.p
                        vtk["velocity"] = particles.v
                        vtk["vorticity"] = vorticity.(particles.∇v)
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
        
        ## Initialize nodal dispacement and pressure with zero
        U = zeros(ndofs(dofmap))
        solved = Sequoia.newton!(U, U->residual(U,state), U->J;
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
    h = sqrt(4*spacing(grid)^2/π)

    ## In the following computation, `@G2P` is unavailable
    ## due to the use of `mpvalues_cell`
    for p in eachindex(particles)
        v̄ₚ = zero(eltype(particles.v))
        mp = mpvalues_cell[whichcell(particles.x[p], grid.X)]
        gridindices = neighboringnodes(mp, grid)
        for ip in eachindex(gridindices)
            i = gridindices[ip]
            v̄ₚ += mp.N[ip] * grid.v[i]
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
        Ṽ[i] = @∑ V[p] * N[ip]
        E[i] = max(0, -V[i] + Ṽ[i])
    end

    E² = sum(E->E^2, grid.E)
    @G2P grid=>i particles=>p mpvalues=>ip begin
        ∇E²[p] = @∑ 2V[p] * E[i] * ∇N[ip]
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
        a[p]  = @∑ a[i] * N[ip]
        p[p]  = @∑ p[i] * N[ip]
        ∇v[p] = @∑ v[i] ⊗ ∇N[ip]
        ∇p[p] = @∑ p[i] * ∇N[ip]
        s[p]  = 2μ * symmetric(∇v[p])
    end

    ## Compute VMS stabilization coefficients based on the current nodal velocity
    compute_VMS_stabilization_coefficients(state)

    ## Compute residual values
    @P2G grid=>i particles=>p mpvalues=>ip begin
        R_mom[i]  = @∑ V[p]*s[p]⋅∇N[ip] - m[p]*b[p]*N[ip] - V[p]*p[p]*∇N[ip] + τ₂[p]*V[p]*tr(∇v[p])*∇N[ip]
        R_mas[i]  = @∑ V[p]*tr(∇v[p])*N[ip] + τ₁[p]*m[p]*(a[p]-b[p])⋅∇N[ip] + τ₁[p]*V[p]*∇p[p]⋅∇N[ip]
        R_mom[i] += m[i]*a[i]
    end

    ## Map grid values to vector `R`
    dofmap(map(vcat, grid.R_mom, grid.R_mas))
end

# ## Jacobian matrix

function jacobian(state)
    (; grid, particles, mpvalues, blockspace, ρ, μ, β, γ, A, dofmap, Δt) = state

    ## Construct the Jacobian matrix
    cₚ = 2μ * one(SymmetricFourthOrderTensor{2})
    I(i,j) = ifelse(i===j, one(Mat{2,2}), zero(Mat{2,2}))
    @threaded @P2G_Matrix grid=>(i,j) particles=>p mpvalues=>(ip,jp) blockspace begin
        A[i,j] = @∑ begin
            Kᵤᵤ = (γ/(β*Δt) * ∇N[ip] ⋅ cₚ ⋅ ∇N[jp]) * V[p] + 1/(β*Δt^2) * I(i,j) * m[p] * N[jp]
            Kᵤₚ = -∇N[ip] * N[jp] * V[p]
            Kₚᵤ = (γ/(β*Δt)) * N[ip] * ∇N[jp] * V[p]
            K̂ᵤᵤ = γ/(β*Δt) * τ₂[p] * ∇N[ip] ⊗ ∇N[jp] * V[p]
            K̂ₚᵤ = 1/(β*Δt^2) * τ₁[p] * ρ * ∇N[ip] * N[jp] * V[p]
            K̂ₚₚ = τ₁[p] * ∇N[ip] ⋅ ∇N[jp] * V[p]
            [Kᵤᵤ+K̂ᵤᵤ           Kᵤₚ
             Mat{1,2}(Kₚᵤ+K̂ₚᵤ) K̂ₚₚ]
        end
    end

    ## Extract the activated degrees of freedom
    submatrix(A, dofmap)
end

using Test                                       #src
if @isdefined(RUN_TESTS) && RUN_TESTS            #src
    @test main(:FLIP) ≈ [0.644,0.259] rtol=0.002 #src
    @test main(:TPIC) ≈ [0.644,0.259] rtol=0.002 #src
end                                              #src
