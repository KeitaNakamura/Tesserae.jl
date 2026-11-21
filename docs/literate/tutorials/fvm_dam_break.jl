# # Finite-volume free-surface flow
#
# This example solves a dam break on a two-dimensional Cartesian MAC grid.
# The 3.22 m by 5 m tank has impermeable, free-slip walls. The initial
# water column is 1.2 m wide and 0.6 m high. Only the liquid is solved, with
# atmospheric pressure at the free surface.
#
# The VTK output is written to `output/fvm_dam_break`.
#
# ```math
# \nabla\cdot\boldsymbol u=0,\qquad
# \partial_t\boldsymbol u+(\boldsymbol u\cdot\nabla)\boldsymbol u
# =-\nabla\pi+\rho^{-1}\nabla\cdot\boldsymbol\tau+\boldsymbol g,\qquad
# \boldsymbol\tau=\mu(\nabla\boldsymbol u+\nabla\boldsymbol u^\mathsf T).
# ```
# The cell centers store the water fraction and kinematic pressure
# ``\pi=p/\rho``. Each velocity component is stored on its normal faces.
#
# [^1]: [Weymouth and Yue (2010), conservative VOF transport](https://doi.org/10.1016/j.jcp.2009.12.018).
# [^2]: [Aniszewski et al. (2021), PARIS free-surface treatment, §3.7.5](https://arxiv.org/html/2012.11744#S3.SS7.SSS5).

# ## Main function

using Tesserae
using Tesserae.Stencil
using LinearAlgebra
import Tesserae: PLIC, VOF

function main()

    ## Simulation parameters
    h = 0.02
    t_stop = 7.0
    width, height = 3.22, 5.0
    water_width, water_height = 1.2, 0.6
    gravity = Vec(0.0, -9.81)
    maximum_step, cfl = 0.002, 0.25
    fps = 30
    if @isdefined(RUN_TESTS) && RUN_TESTS                                   #src
        h = water_width/(11.5+5e-9) # Exercise a cell just above half full. #src
        width, height, t_stop = 32h, 50h, 0.04                              #src
    end                                                                     #src

    ## Material constants
    ρ = 1000.0
    μ = 1.01e-3

    ## Cell and face properties
    CellProp = @NamedTuple begin
        x      :: Vec{2, Float64}
        c      :: Float64
        wet    :: Bool
        p      :: Float64
        div    :: Float64 # Work array for the discrete divergence.
        τxx    :: Float64
        τyy    :: Float64
    end
    FaceProp = @NamedTuple begin
        x     :: Vec{2, Float64}
        v     :: Float64
        δv    :: Float64
        Fw    :: Float64
        α     :: Float64
        layer :: Int
    end

    ## Background grid and initial water fraction
    mesh = CartesianMesh(h, (0, width), (0, height))
    ## At walls, the PLIC normal uses one layer of cells and MC reconstruction uses two layers of face velocities.
    cells = generate_grid(Cell(), CellProp, mesh; halo=1)
    faces = [generate_grid(Face(d), FaceProp, mesh; halo=2) for d in 1:2]
    C = Region(cells)
    cell_indices = CartesianIndices(size(C))
    for I in cell_indices
        cells.c[I] = clamp(water_width/h-(I[1]-1), 0, 1)*clamp(water_height/h-(I[2]-1), 0, 1)
    end

    e = unitoffsets(C)
    ex, ey = e
    face_regions = Region.(faces)
    interior_faces = interior.(face_regions)
    geometry = Ref((cells.c, cells.wet))

    shear = generate_grid(Vertex(), @NamedTuple{x::Vec{2, Float64}, τ::Float64}, mesh)
    V = Region(shear)
    cell_flux, vertex_flux = zeros(C), zeros(V)

    ## Sparse pressure matrix
    A = create_sparse_matrix(C, C, (0ex, -ex, ex, -ey, ey); ndofs=1)
    Ix, Iy = interior_faces
    Lx, Rx = Ix-ex/2, Ix+ex/2
    Ly, Ry = Iy-ey/2, Iy+ey/2
    ax, ay = faces[1].α, faces[2].α
    u, v = faces[1].v, faces[2].v

    ## Output
    outdir = mkpath(joinpath("output", "fvm_dam_break"))
    pvdfile = joinpath(outdir, "paraview")
    closepvd(openpvd(pvdfile)) # Create file
    nframes = max(1, round(Int, t_stop*fps))
    savepoints = collect(range(0.0, t_stop; length=nframes+1)[2:end])
    acceleration = sum(abs, gravity)/h
    viscous_step = ρ*h^2/(8μ)
    time, step = 0.0, 0

    # ## Time integration
    while time < t_stop
        ## Time step from advection, gravity and explicit viscosity.
        rate = sum(maximum(abs, view(face.v, F))/h for (face, F) in zip(faces, face_regions))
        dt = min(maximum_step, viscous_step, 2cfl/(rate+sqrt(rate^2+4acceleration*cfl)), first(savepoints)-time)

        # ### Velocity advection
        #
        # Reconstruct the transported velocity with the MC limiter at each
        # momentum control-volume face. Evaluate both directions at time
        # ``n``. The gradient form is assembled as
        # ``-D(\boldsymbol u u_i)+u_i\overline{D\boldsymbol u}``,
        # where ``D`` is the discrete divergence and the bar averages the two
        # cells adjacent to the velocity node.
        @stencil cells.div[C] = (u[C+ex/2]-u[C-ex/2]+v[C+ey/2]-v[C-ey/2])/h
        for (i, face) in enumerate(faces)
            I, ei = interior_faces[i], e[i]
            fillzero!(face.δv)
            for (d, transport) in enumerate(faces)
                ed = e[d]
                Q, flux = i == d ? (C, cell_flux) : (V, vertex_flux)
                @stencil begin
                    flux[Q] = mc_flux((transport.v[Q-ei/2]+transport.v[Q+ei/2])/2,
                        face.v[Q-3ed/2], face.v[Q-ed/2], face.v[Q+ed/2], face.v[Q+3ed/2])
                    face.δv[I] += (flux[I-ed/2]-flux[I+ed/2])/h
                end
            end
            @stencil face.δv[I] += face.v[I]*(cells.div[I-ei/2]+cells.div[I+ei/2])/2
        end
        ## Viscous stresses at cell centers and vertices.
        @stencil begin
            cells.τxx[C] = 2μ*(u[C+ex/2]-u[C-ex/2])/h
            cells.τyy[C] = 2μ*(v[C+ey/2]-v[C-ey/2])/h
            shear.τ[V] = μ*((u[V+ey/2]-u[V-ey/2])/h+(v[V+ex/2]-v[V-ex/2])/h)
        end

        # ### Water transport
        #
        # PLIC supplies the swept water fraction ``F^w`` at each face.
        # The Weymouth–Yue update[^1] adds the directional compression term
        # ``H(\lambda_+-\lambda_-)``, where ``H=\mathbf{1}_{c^n>1/2}``
        # and ``\lambda=u_d^n\Delta t/h``. The sweep order alternates.
        # Store ``H`` in the wet mask and hold it fixed through both sweeps.
        @stencil cells.wet[C] = cells.c[C] > 0.5
        for sweep in 0:1
            d = mod(step+sweep, 2)+1
            face, ed, factor = faces[d], e[d], dt/h
            F, I = face_regions[d], interior_faces[d]
            L, R = I-ed/2, I+ed/2
            left_wall, right_wall = boundaries(F, d)
            reflect_halos!(cells.c, cells, mesh)
            @stencil face.Fw[I] = VOF.flux(factor*face.v[I], (cells.c,),
                ifelse(face.v[I] > 0, cell_indices[L], cell_indices[R]), d)
            fillzero!(view(face.Fw, [left_wall, right_wall]))
            @stencil cells.c[C] += face.Fw[C-ed/2]-face.Fw[C+ed/2]+
                cells.wet[C]*(factor*face.v[C+ed/2]-factor*face.v[C-ed/2])
        end

        # ### Predictor and pressure projection
        #
        # Add advection, viscosity and gravity to obtain the provisional
        # velocity. At a water–air face, use PLIC to find the water area
        # in the two half-cells next to the face. Fit an interface to their
        # combined water area and intersect it with the line joining the
        # two pressure nodes. If the distance from the water node to the
        # surface is ``\delta h``, the pressure gradient toward air is
        # ``(0-\pi_\mathrm{water})/(\delta h)``[^2].
        #
        # The Poisson matrix and the velocity correction both use the face
        # coefficient ``\alpha=1/\delta``. Between two water pressure nodes
        # use ``\alpha=1``. Between two air nodes use ``\alpha=0``.
        ## Recompute the wet mask from the transported water fraction.
        @stencil cells.wet[C] = cells.c[C] > 0.5
        reflect_halos!(cells.c, cells, mesh)
        for (i, face) in enumerate(faces)
            F, I = face_regions[i], interior_faces[i]
            ei, et, τ = e[i], e[3-i], i == 1 ? cells.τxx : cells.τyy
            fillzero!(face.α)
            @stencil begin
                face.α[I] = interface_weight(geometry, cell_indices[I-ei/2], cell_indices[I+ei/2], i)
                face.v[F] = ifelse(face.α[F] > 0, face.v[F]+dt*(face.δv[F]+gravity[i]+
                    ((τ[F+ei/2]-τ[F-ei/2])/h+(shear.τ[F+et/2]-shear.τ[F-et/2])/h)/ρ), 0)
            end
        end

        ## Use the same face coefficient in the Poisson matrix and velocity correction.
        @stencil cells.div[C] = (u[C+ex/2]-u[C-ex/2]+v[C+ey/2]-v[C-ey/2])/h
        free = DofMap(reshape(view(cells.wet, C), 1, size(C)...))
        rhs = -Array(free(view(cells.div, C)))/dt
        @stencil_assemble begin
            A[C,C] = (ax[C-ex/2]+ax[C+ex/2]+ay[C-ey/2]+ay[C+ey/2])/h^2
            A[Lx,Rx] = -ax[Ix]/h^2
            A[Rx,Lx] = -ax[Ix]/h^2
            A[Ly,Ry] = -ay[Iy]/h^2
            A[Ry,Ly] = -ay[Iy]/h^2
        end
        fillzero!(cells.p)
        free(view(cells.p, C)) .= cholesky(Symmetric(extract(A, free))) \ rhs
        for (d, face) in enumerate(faces)
            I, ed = interior_faces[d], e[d]
            @stencil face.v[I] -= dt*face.α[I]*(cells.p[I+ed/2]-cells.p[I-ed/2])/h
            extend_velocity!(face, mesh, d)
        end

        time += dt
        step += 1

        ## Write results
        if time ≥ first(savepoints)
            popfirst!(savepoints)
            openpvd(pvdfile; append=true) do pvd
                openvtk(string(pvdfile, step), mesh) do vtk
                    vtk["Water fraction"] = cells.c[C]
                    vtk["Pressure (Pa)"] = ρ*cells.p[C]
                    pvd[time] = vtk
                end
            end
        end
    end
    cells, faces #src
end

# ## Local reconstruction and boundary conditions

@inline minmod(a, b) = min(max(a, 0), max(b, 0)) + max(min(a, 0), min(b, 0))

@inline function mc_flux(speed, far_left, left, right, far_right)
    far, up, down = speed > 0 ? (far_left, left, right) : (far_right, right, left)
    a, b = up-far, down-up
    slope = minmod((a+b)/2, minmod(2a, 2b))
    speed*(up+slope/2)
end

function reflect_halos!(field, grid, mesh, component=0)
    for axis in 1:ndims(field), wall in boundaries(mesh, axis)
        H = halos(grid, wall)
        R = reflect(H, wall)
        σ = axis == component ? -1 : 1
        @stencil field[H] = σ*field[R]
    end
end

# Extend the velocity two layers into the air. Each layer averages the
# adjacent values in the preceding layer of the same face grid.
function extend_velocity!(face, mesh, d)
    F = Region(face)
    layers, u = face.layer, face.v
    physical = CartesianIndices(size(F))
    interior_indices = regionindices(u, interior(F, d))[d]
    N = ndims(F)
    offsets = ntuple(N) do axis
        CartesianIndex(ntuple(i -> i == axis, N))
    end

    unfilled = -1
    @stencil layers[F] = ifelse(face.α[F] > 0, 0, unfilled)
    for level in 1:2, I in physical
        layers[I] == unfilled || continue
        sum_velocity, count = 0.0, 0
        for offset in offsets, J in (I-offset, I+offset)
            J in physical || continue
            layers[J] == level-1 || continue
            sum_velocity += u[J]
            count += 1
        end
        count == 0 && continue
        layers[I] = level
        if I[d] in interior_indices
            u[I] = sum_velocity/count
        end
    end
    for wall in boundaries(F, d)
        fillzero!(view(u, wall))
    end
    reflect_halos!(u, face, mesh, d)
end

@inline function interface_weight(fields, L, R, d)
    c, wet = fields
    wet[L] == wet[R] && return Float64(wet[L])
    inv(staggered_distance(c, L, R, d, wet[L]))
end

function staggered_distance(c, L, R, d, wet_left)
    half_step = Vec(d == 1, d == 2)/2
    lo, hi = Vec(-0.5, -0.5), Vec(0.5, 0.5)
    lower_normal, upper_normal = VOF.myc_direction(c, L), VOF.myc_direction(c, R)
    ## Bound the fraction used for geometry, allowing for transport roundoff.
    lower_offset = PLIC.plane_offset(lower_normal, clamp(c[L], 0, 1))
    upper_offset = PLIC.plane_offset(upper_normal, clamp(c[R], 0, 1))
    lower_water = PLIC.box_volume(lower_normal, lower_offset, lo+half_step, hi)
    upper_water = PLIC.box_volume(upper_normal, upper_offset, lo, hi-half_step)
    normal = wet_left ? upper_normal : lower_normal
    if min(0.5-lower_water, 0.5-upper_water) > 0.1
        combined = lower_normal+upper_normal
        normal = combined/sum(abs, combined)
    end
    ## Use one cell width if the pressure line is nearly parallel to the interface.
    abs(normal[d]) > 1e-12 || return 1.0
    offset = PLIC.plane_offset(normal, clamp(lower_water+upper_water, 0, 1))
    ## Keep the distance positive and within one cell width.
    clamp(0.5+(wet_left ? 1 : -1)*offset/normal[d], 0.01, 1.0)
end

using Test                                                              #src
if @isdefined(RUN_TESTS) && RUN_TESTS                                   #src
    cells, faces = main()                                               #src
    C = Region(cells)                                                   #src
    c = cells.c[C]                                                      #src
    @test sum(c)*spacing(cells.x)^2 ≈ 0.72 rtol=64eps(Float64)          #src
    @test minimum(c) ≥ -64eps(Float64) && maximum(c) ≤ 1+64eps(Float64) #src
    @test cells.wet[C] == (c .> 0.5)                                    #src
    x = [cells.x[i, 1][1] for i in axes(c, 1)]                          #src
    initial = clamp.((1.2 .- x)./spacing(cells.x) .+ 0.5, 0, 1)         #src
    shift = sum(x .* c)/sum(c) - sum(x .* initial)/sum(initial)         #src
    @test shift > 0.001 # The water centroid moves right by over 1 mm.  #src
end                                                                     #src
