# # Marker-and-cell (MAC) method

using Tesserae
using Tesserae.Stencil # Staggered grid regions

using Krylov: cg
using LinearOperators: LinearOperator

function main()

    ## Simulation parameters
    dim = 3
    h = 0.0322    # grid spacing
    t_stop = 10.0 # final time
    Δt = 2.0e-3   # time step
    g = 9.81      # gravity magnitude

    ## Material constants
    ρ = 1.0e3   # density
    μ = 1.01e-3 # dynamic viscosity
    ν = μ / ρ   # kinematic viscosity

    ## Properties
    CellProp = @NamedTuple begin
        x    :: Vec{dim, Float64} # cell-center position
        V    :: Float64           # volume
        ϕ    :: Float64           # volume fraction (V / h^dim)
        φ    :: Float64           # implicit free-surface field
        φtmp :: Float64           # temporary field for surface smoothing
        p    :: Float64           # pressure
        δp   :: Float64           # temporary vector for matvec (A*x)
        Δp   :: Float64           # Laplacian
        q    :: Float64           # divergence field
    end
    FaceProp = @NamedTuple begin
        x    :: Vec{dim, Float64} # face-center position
        m    :: Float64           # mass
        mv   :: Float64           # momentum component (along face axis)
        v    :: Float64           # face velocity component
        ∇p   :: Float64           # pressure gradient component on faces
        invθ :: Float64           # inverse fluid-to-surface cut fraction
        Δv   :: Float64           # Laplacian for viscosity
        b    :: Float64           # body force component (e.g., gravity)
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
    domain = ((0,3.22), (0,0.5), (0,2.5))
    mesh = CartesianMesh(h, domain...)
    paddedmesh = CartesianMesh(h, domain...; pad)

    cells = generate_grid(CellProp, macmesh(paddedmesh, Cell()))
    facegrids = ntuple(d -> generate_grid(FaceProp, macmesh(paddedmesh, Face(d))), dim)

    fluid = falses(size(cells))

    Ωc = Region(Cell(), physical, physical, physical; halowidth=pad)
    Ωf = ntuple(d -> Region(Face(d), physical, physical, physical; halowidth=pad), dim)

    ## Gravity acts on the vertical velocity component (here: d=3)
    fill!(view(facegrids[3].b, Ωf[3]), -g)

    ## Particles (sampled in the domain, then filtered to the fluid region)
    particles = generate_particles(ParticleProp, paddedmesh; alg=PoissonDiskSampling())
    particles.V .= volume(paddedmesh) / length(particles)
    filter!(pt -> isinside(pt.x, mesh) && 0<pt.x[1]<1.2 && 0<pt.x[3]<0.6, particles)
    @. particles.m = ρ * particles.V
    @show length(particles)

    ## Basis weights
    cellweight = generate_basis_weights(BSpline(Constant()), cells.x, length(particles))
    faceweights = map(grid -> generate_basis_weights(BSpline(Linear()), grid.x, length(particles); derivative=Order(1)), facegrids)

    ## Output
    outdir = mkpath(joinpath("output", "macgrid"))
    pvdfile = joinpath(outdir, "paraview")
    closepvd(openpvd(pvdfile)) # Create file

    t = 0.0
    step = 0
    fps = 30
    savepoints = collect(LinRange(t, t_stop, round(Int, t_stop*fps)+1))

    Tesserae.@showprogress while t < t_stop

        ## P2G: transfer volume to cells, then compute volume fraction ϕ
        update!(cellweight, particles, cells.x)
        @P2G cells=>i particles=>p cellweight=>ip begin
            V[i] = @∑ w[ip] * V[p]
        end
        for axis in 1:dim, halo_axis in (lowhalo, highhalo)
            halo = wallregion(Cell(), halo_axis, axis; halowidth=pad)
            fold!(cells.V, halo, axis, +1)
        end
        fillzero!(cells.ϕ)
        @views @. cells.ϕ[Ωc] = cells.V[Ωc] / h^dim

        ## Reconstruct an implicit free surface from marker occupancy
        reconstructsurface!(cells.φ, cells.φtmp, cells.V, Ωc; halowidth=pad)
        inversecutfractions!(getproperty.(facegrids, :invθ), cells.φ, Ωf)

        ## P2G: transfer mass/momentum to faces, update face velocities, add viscosity+body force
        for d in 1:dim
            grid = facegrids[d]
            weights = faceweights[d]
            Ω = Ωf[d]
            update!(weights, particles, grid.x)

            ## mass & momentum transfer
            @P2G grid=>i particles=>p weights=>ip begin
                m[i]  = @∑ w[ip] * m[p]
                mv[i] = @∑ w[ip] * m[p] * (v[p][d] + ∇v[p][d,:] ⋅ (x[i] - x[p]))
            end
            for axis in 1:dim, halo_axis in (lowhalo, highhalo)
                halo = wallregion(Face(d), halo_axis, axis; halowidth=pad)
                sign = axis == d ? -1 : +1
                fold!(grid.m, halo, axis, +1)
                fold!(grid.mv, halo, axis, sign)
            end
            for boundary_axis in (lowboundary, highboundary)
                boundary = wallregion(Face(d), boundary_axis, d; halowidth=pad)
                view(grid.m, boundary) .*= 2
                fill!(view(grid.mv, boundary), 0)
            end

            # Face velocity component
            @views @. grid.v[Ω] = grid.mv[Ω] / grid.m[Ω] * !iszero(grid.m[Ω])
            freeslip!(grid.v, d; halowidth=pad)

            # Viscous term: ν ∇²v, then explicit update with body force
            laplacian!(grid.Δv, grid.v, Ω; spacing=h)
            @views @. grid.v[Ω] += Δt * (ν * grid.Δv[Ω] + grid.b[Ω])
            freeslip!(grid.v, d; halowidth=pad)
        end

        ## Compute divergence of face velocities at cell centers
        divergence!(cells.q, getproperty.(facegrids, :v), Ωc; spacing=h)

        ## Setup fluid region using the reconstructed free surface
        fill!(fluid, false)
        @views @. fluid[Ωc] = cells.φ[Ωc] < 0
        dofmap = DofMap(reshape(fluid, 1, size(cells)...))

        ## Solve Poisson for pressure on active (fluid) cells, then mirror pressure to padded region
        b = -dofmap(cells.q)
        A = pressurelaplacian(; cells, facegrids, dofmap, Ωc, Ωf, pad, spacing=h, ρ, Δt)

        fillzero!(cells.p)
        dofmap(cells.p) .= cg(A, b; verbose=0)[1]
        for axis in 1:dim, halo_axis in (lowhalo, highhalo)
            halo = wallregion(Cell(), halo_axis, axis; halowidth=pad)
            reflect!(cells.p, halo, axis, +1)
        end

        ## Compute pressure gradient on faces
        gradient!(getproperty.(facegrids, :∇p), cells.p, getproperty.(facegrids, :invθ), Ωf; spacing=h)

        fillzero!(particles.v)
        fillzero!(particles.∇v)
        for d in 1:dim
            grid = facegrids[d]
            weights = faceweights[d]
            Ω = Ωf[d]

            ## Divergence-free projection
            @views @. grid.v[Ω] -= Δt/ρ * grid.∇p[Ω]
            freeslip!(grid.v, d; halowidth=pad)

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
            isinside(x_new, mesh) ? x_new : x
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
function pressurelaplacian(; cells, facegrids, dofmap, Ωc, Ωf, pad, spacing, ρ, Δt)
    pressuregradient = getproperty.(facegrids, :∇p)
    invθ = getproperty.(facegrids, :invθ)
    function mul!(y, x)
        fillzero!(cells.δp)
        dofmap(cells.δp) .= x
        for axis in 1:3, halo_axis in (lowhalo, highhalo)
            halo = wallregion(Cell(), halo_axis, axis; halowidth=pad)
            reflect!(cells.δp, halo, axis, +1)
        end
        gradient!(pressuregradient, cells.δp, invθ, Ωf; spacing)
        divergence!(cells.Δp, pressuregradient, Ωc; spacing)
        y .= (-Δt/ρ) .* dofmap(cells.Δp)
    end
    LinearOperator(Float64, ndofs(dofmap), ndofs(dofmap), true, true, mul!)
end

function reconstructsurface!(φ, φtmp, volume, Ω; halowidth)
    @views @. φ[Ω] = ifelse(volume[Ω] > 0, -1.0, +1.0)
    smoothimplicit!(φ, φtmp, Ω; halowidth)
    for axis in 1:3, halo_axis in (lowhalo, highhalo)
        halo = wallregion(Cell(), halo_axis, axis; halowidth)
        reflect!(φ, halo, axis, +1)
    end
    φ
end

function smoothimplicit!(φ, φtmp, Ω; halowidth)
    # Apply one separable [1, 2, 1] weighted average.
    for (axis, offset) in enumerate(unitoffsets(Val(3)))
        for halo_axis in (lowhalo, highhalo)
            halo = wallregion(Cell(), halo_axis, axis; halowidth)
            reflect!(φ, halo, axis, +1)
        end
        @views @. φtmp[Ω] = (φ[Ω - offset] + 2 * φ[Ω] + φ[Ω + offset]) / 4
        copyto!(view(φ, Ω), view(φtmp, Ω))
    end
    φ
end

function inversecutfractions!(invθ, φ, Ωf)
    e₁, e₂, e₃ = unitoffsets(Val(3))
    invθ₁, invθ₂, invθ₃ = invθ
    F₁, F₂, F₃ = Ωf
    @views @. invθ₁[F₁] = inversecutfraction(φ[F₁ - e₁/2], φ[F₁ + e₁/2])
    @views @. invθ₂[F₂] = inversecutfraction(φ[F₂ - e₂/2], φ[F₂ + e₂/2])
    @views @. invθ₃[F₃] = inversecutfraction(φ[F₃ - e₃/2], φ[F₃ + e₃/2])
    invθ
end

function inversecutfraction(φ₋, φ₊)
    fluid₋ = φ₋ < 0
    fluid₊ = φ₊ < 0
    fluid₋ == fluid₊ && return 1.0
    φfluid = fluid₋ ? φ₋ : φ₊
    φair = fluid₋ ? φ₊ : φ₋
    θ = φfluid / (φfluid - φair)
    inv(max(θ, 1.0e-6))
end

function macmesh(mesh::CartesianMesh{3}, location)
    h = spacing(mesh)
    firstnode = mesh[1, 1, 1]
    nodes = ntuple(axis -> range(firstnode[axis]; step=h, length=size(mesh, axis)), 3)
    centers = map(axis -> range(first(axis) + h/2; step=h, length=length(axis)-1), nodes)
    coordinateaxes = ntuple(axis -> location == Face(axis) ? nodes[axis] : centers[axis], 3)
    CartesianMesh(map(collect, coordinateaxes), h, inv(h))
end

function laplacian!(Δu, u, Ω; spacing)
    e₁, e₂, e₃ = unitoffsets(Val(3))
    @views @. Δu[Ω] = (
        u[Ω - e₁] + u[Ω + e₁] +
        u[Ω - e₂] + u[Ω + e₂] +
        u[Ω - e₃] + u[Ω + e₃] - 6u[Ω]
    ) / spacing^2
    Δu
end

function divergence!(q, velocity, Ω; spacing)
    e₁, e₂, e₃ = unitoffsets(Val(3))
    u, v, w = velocity
    @views @. q[Ω] = (
        u[Ω + e₁/2] - u[Ω - e₁/2] +
        v[Ω + e₂/2] - v[Ω - e₂/2] +
        w[Ω + e₃/2] - w[Ω - e₃/2]
    ) / spacing
    q
end

function gradient!(∇p, p, invθ, Ωf; spacing)
    e₁, e₂, e₃ = unitoffsets(Val(3))
    ∇p₁, ∇p₂, ∇p₃ = ∇p
    invθ₁, invθ₂, invθ₃ = invθ
    F₁, F₂, F₃ = Ωf
    @views @. ∇p₁[F₁] = invθ₁[F₁] * (p[F₁ + e₁/2] - p[F₁ - e₁/2]) / spacing
    @views @. ∇p₂[F₂] = invθ₂[F₂] * (p[F₂ + e₂/2] - p[F₂ - e₂/2]) / spacing
    @views @. ∇p₃[F₃] = invθ₃[F₃] * (p[F₃ + e₃/2] - p[F₃ - e₃/2]) / spacing
    ∇p
end

function wallregion(location, axisregion::AxisRegion, axis::Int; halowidth::Int)
    axisregions = ntuple(dimension -> dimension == axis ? axisregion : full, 3)
    Region(location, axisregions; halowidth)
end

function fold!(A, halo, axis::Int, sign::Int)
    @views @. A[reflect(halo, axis)] += sign * A[halo]
    fill!(view(A, halo), 0)
    A
end

function reflect!(A, halo, axis::Int, sign::Int)
    @views @. A[halo] = sign * A[reflect(halo, axis)]
    A
end

function freeslip!(A, d::Int; halowidth::Int)
    for axis in 1:3, halo_axis in (lowhalo, highhalo)
        halo = wallregion(Face(d), halo_axis, axis; halowidth)
        sign = axis == d ? -1 : +1
        reflect!(A, halo, axis, sign)
    end
    for boundary_axis in (lowboundary, highboundary)
        boundary = wallregion(Face(d), boundary_axis, d; halowidth)
        fill!(view(A, boundary), 0)
    end
    A
end
