# Interface geometry and volume-of-fluid kernels

`Tesserae.PLIC` and `Tesserae.VOF` provide local calculations for two- and
three-dimensional Cartesian cells. The [free-surface tutorial](tutorials/fvm_dam_break.md)
uses them to transport water on a MAC grid.
The modules allocate no field arrays and prescribe no boundary conditions,
time step, material data or pressure solver.

## Plane and box geometry

A plane with normal ``\boldsymbol{n}`` and offset ``a`` defines the filled side
``\boldsymbol{n}\cdot\boldsymbol{x}\leq a``. The reference cell is
``[-1/2,1/2]^N``. Normals and box corners are `Vec` values. Normals must be
finite and nonzero but need not be normalized. Use the same floating-point
type for the normal, offset or fraction, and box corners.
Float32 and Float64 inputs retain their precision.

```jldoctest
julia> using Tesserae: PLIC, Vec

julia> n = Vec(1.0, 0.0);

julia> PLIC.plane_fraction(n, 0.0)
0.5

julia> PLIC.plane_offset(n, 0.25)
-0.25

julia> PLIC.box_volume(n, 0.0, Vec(-0.5, -0.5), Vec(0.0, 0.5))
0.5
```

`plane_fraction` returns a fraction of the reference cell; `box_volume` returns
the absolute area or volume inside the supplied sub-box. The geometric inverse
`PLIC.plane_offset` requires a fraction in ``[0,1]``.

```@docs
Tesserae.PLIC.plane_fraction
Tesserae.PLIC.plane_offset
Tesserae.PLIC.box_volume
```

## Interface reconstruction

`myc_normal(fraction, I)` estimates the MYC normal at an ordinary
`CartesianIndex`. It reads one surrounding layer for mixed cells; a pure center
uses the positive-x convention because there is no local interface to reconstruct.
Both estimators return `Vec` normals.
`myc_direction` also reads the neighbors at pure centers, as needed by some
free-surface pressure discretizations. The caller fills any required halo.

The normal uses reference-cell coordinates. On a rectangular grid,
`normal ./ spacing` gives its physical direction. A normal estimator can be
replaced independently of the PLIC geometry and donor flux.

```@docs
Tesserae.VOF.myc_normal
Tesserae.VOF.myc_direction
```

Use `PLIC.plane_offset(n, c)` to locate the plane for a fraction in `[0, 1]`.
If the transport calculation can leave small finite excursions outside this
interval, the caller can pass `clamp(c, zero(c), one(c))` to the geometric
inverse. The stored fraction still enters the transport balance directly.

## Geometric donor flux

The signed displacement is ``\lambda=u\Delta t/h``. `VOF.flux` computes the
filled volume swept from one selected donor cell, divided by that whole cell's
volume. Its sign chooses the donor's positive or negative face. The caller
selects the donor and writes the face-flux difference in the conservation law.
Pass the donor fraction, normal and offset for a precomputed plane, or pass
the fraction field and donor index to reconstruct that cell with MYC.
The geometric kernel supports ``|\lambda|\leq1``; a complete transport scheme
can impose a stricter CFL bound. The dam-break tutorial uses 0.25.

```@docs
Tesserae.VOF.flux
```

The scalar kernels can be used in native broadcast and `@stencil` expressions
on CPU and GPU arrays. This does not select a GPU pressure solver for the
complete tutorial.
