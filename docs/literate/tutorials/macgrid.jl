# # Marker-and-cell (MAC) method

using Tesserae
using Tesserae.Stencil # Staggered grid types and stencil operators (MAC layout)

using Krylov: cg
using LinearOperators: LinearOperator

function main()

    ## Simulation parameters
    dim = 3
    h = 0.0322  # grid spacing
    T = 10.0    # final time
    Δt = 2.0e-3 # time step
    g = 9.81    # gravity magnitude

    ## Material constants
    ρ = 1.0e3   # density
    μ = 1.01e-3 # dynamic viscosity
    ν = μ / ρ   # kinematic viscosity

    ## Properties
    CellProp = @NamedTuple begin
        x  :: Vec{dim, Float64} # cell-center position
        V  :: Float64           # volume
        ϕ  :: Float64           # volume fraction (V / h^dim)
        p  :: Float64           # pressure
        δp :: Float64           # temporary vector for matvec (A*x)
        Δp :: Float64           # Laplacian
        q  :: Float64           # divergence field
    end
    FaceProp = @NamedTuple begin
        x  :: Vec{dim, Float64} # face-center position
        m  :: Float64           # mass
        mv :: Float64           # momentum component (along face axis)
        v  :: Float64           # face velocity component
        ∇p :: Float64           # pressure gradient component on faces
        Δv :: Float64           # Laplacian for viscosity
        b  :: Float64           # body force component (e.g., gravity)
    end
    ParticleProp = @NamedTuple begin
        x   :: Vec{dim, Float64}                        # position
        m   :: Float64                                  # mass
        V   :: Float64                                  # volume
        v   :: Vec{dim, Float64}                        # velocity
        ∇v  :: SecondOrderTensor{dim, Float64, dim*dim} # velocity gradient (APIC-style)
        vᵈ  :: Float64                                  # scratch: interpolated face component
        ∇vᵈ :: Vec{dim, Float64}                        # scratch: grad of interpolated component
        p   :: Float64                                  # pressure
    end

    ## Background grid (staggered/MAC: pressure at cells, velocity components at faces)
    pad = 1
    mesh = CartesianMesh(h, (0,3.22), (0,0.5), (0,2.5); pad)

    cells = generate_grid(Cell(), CellProp, mesh)
    facegrids = ntuple(d -> generate_grid(Face(d), FaceProp, mesh), dim)

    ## Gravity acts on the vertical velocity component (here: d=3)
    inner(facegrids[3].b; pad, dropboundary=true) .= -g

    ## Particles (sampled in the domain, then filtered to the fluid region)
    particles = generate_particles(ParticleProp, mesh; alg=PoissonDiskSampling())
    particles.V .= volume(mesh) / length(particles)
    filter!(pt -> isinside(pt.x, inner(mesh; pad)) && 0<pt.x[1]<1.2 && 0<pt.x[3]<0.6, particles)
    @. particles.m = ρ * particles.V
    @show length(particles)

    ## Interpolation weights
    cellweight = generate_interpolation_weights(BSpline(Constant()), cells.x, length(particles))
    faceweights = map(grid -> generate_interpolation_weights(BSpline(Linear()), grid.x, length(particles); derivative=Order(1)), facegrids)

    ## Output
    outdir = mkpath(joinpath("output", "macgrid"))
    pvdfile = joinpath(outdir, "paraview")
    closepvd(openpvd(pvdfile)) # Create file

    t = 0.0
    step = 0
    fps = 30
    savepoints = collect(LinRange(t, T, round(Int, T*fps)+1))

    Tesserae.@showprogress while t < T

        ## P2G: transfer volume to cells, then compute volume fraction ϕ
        update!(cellweight, particles, cells.x)
        @P2G cells=>i particles=>p cellweight=>ip begin
            V[i] = @∑ w[ip] * V[p]
        end
        foldpad!(cells.V; pad)
        for d in 1:dim
            Stencil.outer(cells.V, d; pad) .= 0
        end
        @. cells.ϕ = cells.V / h^dim

        ## P2G: transfer mass/momentum to faces, update face velocities, add viscosity+body force
        for d in 1:dim
            grid = facegrids[d]
            weights = faceweights[d]
            update!(weights, particles, grid.x)

            ## mass & momentum transfer
            @P2G grid=>i particles=>p weights=>ip begin
                m[i]  = @∑ w[ip] * m[p]
                mv[i] = @∑ w[ip] * m[p] * (v[p][d] + ∇v[p][d,:] ⋅ (x[i] - x[p]))
            end
            foldpad!(grid.m; pad)
            foldpad!(grid.mv; pad, flip = (d .== 1:dim)) # flip: consider slip boundary

            # Face velocity component
            @. grid.v = grid.mv / grid.m * !iszero(grid.m)
            mirrorpad!(grid.v; pad, flip = (d .== 1:dim)) # flip: consider slip boundary

            # Viscous term: ν ∇²v, then explicit update with body force
            stencil!(Laplacian(), grid.Δv, grid.v; pad, spacing=h)
            @. grid.v += Δt * (ν * grid.Δv + grid.b)
            mirrorpad!(grid.v; pad, flip = (d .== 1:dim))
        end

        ## Compute divergence of face velocities at cell centers
        stencil!(Divergence(), cells.q, getproperty.(facegrids, :v); pad, spacing=h)

        ## Setup fluid region using volume fraction
        mask = cells.ϕ .> 0  # active fluid region
        dofmap = DofMap(reshape(mask, 1, size(cells)...))

        ## Solve Poisson for pressure on active (fluid) cells, then mirror pressure to padded region
        b = -dofmap(cells.q)
        A = laplacian(; facegrids, cells, dofmap, pad, spacing=h, ρ, Δt)

        fillzero!(cells.p)
        dofmap(cells.p) .= cg(A, b; verbose=0)[1]
        mirrorpad!(cells.p; pad)

        ## Compute pressure gradient on faces
        stencil!(Gradient(), getproperty.(facegrids, :∇p), cells.p; pad, spacing=h)

        fillzero!(particles.v)
        fillzero!(particles.∇v)
        for d in 1:dim
            grid = facegrids[d]
            weights = faceweights[d]

            ## Divergence-free projection
            @. grid.v -= Δt/ρ * grid.∇p
            mirrorpad!(grid.v; pad, flip = (d .== 1:dim))

            ## Interpolate corrected face component back to particles (and its gradient)
            e = Vec{dim}(==(d)) # unit vector for component d
            @G2P grid=>i particles=>p weights=>ip begin
                vᵈ[p] = @∑ v[i] * w[ip]
                ∇vᵈ[p] = @∑ v[i] * ∇w[ip]
                v[p] += e * vᵈ[p]
                ∇v[p] += e ⊗ ∇vᵈ[p]
            end
        end
        function update_position(x, v)
            x_new = x + v*Δt
            isinside(x_new, inner(mesh; pad)) ? x_new : x
        end
        @. particles.x = update_position(particles.x, particles.v)

        t += Δt
        step += 1

        ## Write results
        if t > first(savepoints)
            popfirst!(savepoints)
            openpvd(pvdfile; append=true) do pvd
                @G2P cells=>i particles=>p cellweight=>ip begin
                    p[p] = @∑ w[ip] * p[i]
                end
                openvtm(string(pvdfile, step)) do vtm
                    openvtk(vtm, particles.x) do vtk
                        vtk["Velocity (m/s)"] = particles.v
                        vtk["Pressure (Pa)"] = particles.p
                    end
                    openvtk(vtm, cells.x) do vtk
                        vtk["Pressure (Pa)"] = cells.p
                        vtk["Volume fraction"] = cells.ϕ
                    end
                    pvd[t] = vtm
                end
            end
        end
    end
    sum(particles.x) / length(particles) #src
end

## Matrix-free Laplacian operator for pressure Poisson (restricted to dofmap)
function laplacian(; facegrids, cells, dofmap, pad, spacing, ρ, Δt)
    dim = length(facegrids)
    function mul!(y, x)
        fillzero!(cells.δp)
        dofmap(cells.δp) .= x                                   # set `x` into cell field
        mirrorpad!(cells.δp; pad)                               # set ghost values for boundary conditions (no flux)
        stencil!(Laplacian(), cells.Δp, cells.δp; pad, spacing) # compute Laplacian
        @. y = -Δt / ρ * $dofmap(cells.Δp) # gather back to DOF vector (A*x)
    end
    LinearOperator(Float64, ndofs(dofmap), ndofs(dofmap), true, true, mul!)
end
