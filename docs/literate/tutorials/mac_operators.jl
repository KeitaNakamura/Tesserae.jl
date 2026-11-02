# # MAC operators with wall and periodic boundaries
#
# Pressure gradients, divergence, and velocity Laplacians on a staggered grid.
# This example defines the discrete operators; it does not solve a time step.
# Scalars, regions, allocations, and standalone helper calls are prepared outside each
# `@stencil` block. Blocks group array updates, executed in the written order;
# every equation independently selects its valid calculation points.

using Tesserae.Stencil

"""
Differentiate `a`, centered along axis `d`, at the grid-line positions in `R`.
Set the interior differences, periodic seam, and duplicate boundary explicitly.
This helper is local to the example.
"""
function periodic_difference!(out, a, R, d, hd)
    e = unitoffsets(R)[d]
    L, U = boundaries(R, d)
    @stencil begin
        out[R] = (a[R+e/2] - a[R-e/2])/hd
        out[L] = (a[L+e/2] - a[U-e/2])/hd
        out[U] = out[L]
    end
    out
end

"""
MAC operators for ordinary arrays: cell-centered pressure `p`, x-face velocity `u`,
and y-face velocity `v`. Apply no-slip walls and zero normal pressure gradients.
Update boundary-normal velocities in place; no pressure Poisson solve is performed.
"""
function wall_operators(p, u, v, h)
    C = Region(Cell(), size(p))
    V = Region(Vertex(), C)
    faces = [Region(Face(d), C) for d in 1:2]
    offsets = unitoffsets(C)
    velocity = [u, v]
    gradient = [similar(p, size(F)) for F in faces]
    normal = [similar(p) for _ in velocity]
    transverse = [similar(p, size(V)) for _ in velocity]
    laplacian = [similar(a) for a in velocity]
    divergence = similar(p)
    fill!(divergence, 0)

    # d=1 pairs u with x; d=2 pairs v with y. j is the transverse axis.
    for (d, e) in enumerate(offsets)
        j = 3-d
        et, hd, ht = offsets[j], h[d], h[j]
        F = faces[d]
        L, R = boundaries(F)
        a, g, dn, ds, lap = velocity[d], gradient[d], normal[d], transverse[d], laplacian[d]

        # Normal wall velocity and normal pressure gradient are zero.
        @stencil begin
            a[L] = 0
            a[R] = 0
            g[F] = (p[F+e/2] - p[F-e/2])/hd
            g[L] = 0
            g[R] = 0

            # Accumulate this direction's contribution to divergence.
            dn[C] = (a[C+e/2] - a[C-e/2])/hd
            divergence[C] += dn[C]
        end

        # Enforce tangential no-slip using the half-cell distance to the wall.
        B, T = boundaries(V, j)
        @stencil begin
            ds[V] = (a[V+et/2] - a[V-et/2])/ht
            ds[B] =  a[B+et/2]/(ht/2)
            ds[T] = -a[T-et/2]/(ht/2)
        end

        fill!(lap, 0)
        @stencil lap[F] =
            (dn[F+e/2] - dn[F-e/2])/hd + (ds[F+et/2] - ds[F-et/2])/ht
    end

    (; gx=gradient[1], gy=gradient[2], divergence,
       du_dx=normal[1], dv_dy=normal[2], du_dy=transverse[1], dv_dx=transverse[2],
       lap_u=laplacian[1], lap_v=laplacian[2])
end

"""
MAC operators with periodic boundaries in both directions.
The last x-face of `u` and last y-face of `v` duplicate the corresponding first
faces and are synchronized before differencing.
"""
function periodic_operators(p, u, v, h)
    C = Region(Cell(), size(p))
    V = Region(Vertex(), C)
    faces = [Region(Face(d), C) for d in 1:2]
    offsets = unitoffsets(C)
    velocity = [u, v]
    gradient = [similar(p, size(F)) for F in faces]
    normal = [similar(p) for _ in velocity]
    transverse = [similar(p, size(V)) for _ in velocity]
    laplacian = [similar(a) for a in velocity]
    divergence = similar(p)
    fill!(divergence, 0)

    for (d, e) in enumerate(offsets)
        j = 3-d
        et, hd, ht = offsets[j], h[d], h[j]
        F = faces[d]
        L, R = boundaries(F)
        a, g, dn, ds, lap = velocity[d], gradient[d], normal[d], transverse[d], laplacian[d]

        @stencil a[R] = a[L]
        periodic_difference!(g, p, F, d, hd)

        @stencil begin
            dn[C] = (a[C+e/2] - a[C-e/2])/hd
            divergence[C] += dn[C]
        end

        periodic_difference!(ds, a, V, j, ht)
        periodic_difference!(lap, dn, F, d, hd)
        @stencil lap[F] += (ds[F+et/2] - ds[F-et/2])/ht
    end

    (; gx=gradient[1], gy=gradient[2], divergence,
       du_dx=normal[1], dv_dy=normal[2], du_dy=transverse[1], dv_dx=transverse[2],
       lap_u=laplacian[1], lap_v=laplacian[2])
end

function mac_example(n=(32, 24))
    C = Region(Cell(), n)
    Fx, Fy = Region(Face(1), C), Region(Face(2), C)
    h = 1 ./ n

    # Wall case: smooth fields with zero normal pressure gradient and zero wall velocity.
    p = [cospi(x)*cospi(y) for (x, y) in coordinates(C, h)]
    u = [x*(1-x)*y*(1-y) for (x, y) in coordinates(Fx, h)]
    v = [2*x*(1-x)*y*(1-y) for (x, y) in coordinates(Fy, h)]
    walls = wall_operators(p, u, v, h)

    # Periodic case: matching endpoint values from trigonometric fields.
    pp = [sinpi(2*x)+0.5*sinpi(2*y) for (x, y) in coordinates(C, h)]
    up = [cospi(2*x)*cospi(2*y) for (x, y) in coordinates(Fx, h)]
    vp = [sinpi(2*x)*cospi(2*y) for (x, y) in coordinates(Fy, h)]
    periodic = periodic_operators(pp, up, vp, h)

    # Net boundary flux vanishes for both impermeable and periodic boundaries.
    println("cells: ", n)
    println("wall net flux: ", sum(walls.divergence)*prod(h))
    println("periodic net flux: ", sum(periodic.divergence)*prod(h))
    (; walls, periodic)
end

if abspath(PROGRAM_FILE) == @__FILE__
    mac_example()
end
