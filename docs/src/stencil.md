# Staggered grids and stencil equations

`Tesserae.Stencil` lets you write Cartesian finite differences and volume balances
using regions of cells, faces, edges, or vertices. You choose the field placement,
boundary equations, and time update. Stencil maps the regions to the underlying arrays.

## Field placement and regions

Use the same property declaration as for [MPM grids](generation.md), with a placement
and the number of halo layers for the new grid.

```jldoctest stencil_grid
julia> using Tesserae, Tesserae.Stencil

julia> mesh = CartesianMesh(0.25, (0, 1), (0, 1));

julia> Prop = @NamedTuple{x::Vec{2, Float64}, v::Float64};

julia> grid = generate_grid(Face(1), Prop, mesh; halo=1);

julia> F = Region(grid);

julia> size(F), size(grid)
((5, 4), (7, 6))
```

`Cell()` places samples at cell centers and `Face(d)` at the centers of faces normal
to direction `d`. In three dimensions, `Edge(d)` selects edges parallel to direction
`d`. For example, `Edge(3)` is the placement for the `xy` shear stress on a MAC grid.
`Vertex()` selects grid vertices.

`Region(grid)` selects the physical samples. The field arrays also store halos.
`Region(location, mesh)` makes the same selection without allocating a field.
`Region(location, C)` changes the placement on the same physical domain, while
`Region(location, counts)` takes the sample counts at the requested placement.
For example, four cells in the first direction give five samples at `Face(1)`.
Placement conversion and field allocation require a whole-domain region.
Use shifts and intersections to select neighboring samples within that domain.

With `ex, ey = unitoffsets(C)`, `ex` is a shift of one cell in the first direction
and `C+ex/2` selects the corresponding faces. `coordinates(R)` iterates over the
points in selection order. A logical region can also use
`coordinates(R, (hx, hy); origin=(x0, y0))` to supply a physical scale.
Use `A[R]` or `view(A, R)` to access values. For a single field,
`zeros(T, R; halo=1)` allocates CPU storage and `similar(A, T, R; halo=1)`
allocates on the same backend as `A`.
`fill(value, R; halo=1)` allocates a field with a prescribed initial value.
Placement expressions such as `Face(1)+ey/2` can also be passed to
`generate_grid`; the offset changes which axes are aligned with grid lines.

`regionindices(A, R)` returns the array index ranges underlying `A[R]`.
It preserves the array's index origin and the region's traversal order.
It does not clip the selection to storage; normal indexing checks bounds,
while `@stencil` applies the common-range rule described below.

```jldoctest
julia> using Tesserae.Stencil

julia> C = Region(Cell(), (4,)); (ex,) = unitoffsets(C);

julia> A = zeros(C; halo=1);

julia> map(first, regionindices(A, C)), map(last, regionindices(A, C))
((1,), (4,))

julia> map(first, regionindices(A, C-ex)), map(last, regionindices(A, C-ex))
((0,), (3,))
```

```@docs
Tesserae.Stencil.Cell
Tesserae.Stencil.Face
Tesserae.Stencil.Edge
Tesserae.Stencil.Vertex
Tesserae.Stencil.Region
Tesserae.Stencil.unitoffsets
Tesserae.Stencil.coordinates
Tesserae.Stencil.regionindices
Base.zeros(::Type{T}, ::Tesserae.Stencil.Region) where {T}
Base.fill(::Any, ::Tesserae.Stencil.Region)
Base.similar(::AbstractArray, ::Type{T}, ::Tesserae.Stencil.Region) where {T}
Base.:+(::Tesserae.Stencil.Location, ::Tesserae.Stencil.GridOffset{N}) where {N}
```

## Boundary selections

`boundaries(mesh, d)` returns the two wall planes for direction `d`.
`boundaries(F, d)` selects the samples of `F` on those walls. For face regions,
`boundaries(F)` uses the face normal, and `interior(F)` selects faces inside the
domain. `interior(F, d)` restricts the selection along just one direction.

```julia
CellProp = @NamedTuple{x::Vec{2, Float64}, p::Float64}
cells = generate_grid(Cell(), CellProp, mesh; halo=1)
wall_pressure = 0.0
left, right = boundaries(mesh, 1)
H = halos(cells, left)
I = reflect(H, left)
@stencil cells.p[H] = 2wall_pressure - cells.p[I]

L, R = boundaries(F)
@stencil grid.v[[L, R]] = 0
```

The pressure example uses a cell-centered field, where reflection pairs each
halo sample with an interior sample in reverse order along the wall normal.
`halos(A, C, left)` also works with a standalone array and its physical region.
It uses the same physical sample mapping as `A[C]`. Arrays with no extra
storage have empty halo selections, including when their axes are offset.
Halo selections include corners along the transverse directions.

`AxisPlane(mesh, d, x)` places an axis-aligned plane at physical coordinate
`x`. `intersect(R, plane)` selects samples on it and `split(R, plane)` returns
the samples on the negative side, on the plane, and on the positive side.
Reflection requires the image to land on the sampling lattice.

The grid-line samples retain the coordinates supplied by the input mesh,
including `LinRange` axes and mesh slices. Half-cell positions and samples
outside the input axes use the mesh origin and spacing. Regions share the
input coordinate axes; treat mesh geometry as read-only after creating them.

```@docs
Tesserae.Stencil.interior
Tesserae.boundaries(::Tesserae.Stencil.Region, ::Int)
Tesserae.boundaries(::Tesserae.CartesianMesh, ::Int)
Tesserae.Stencil.AxisPlane
Tesserae.Stencil.reflect
Tesserae.Stencil.halos
Base.intersect(::Tesserae.Stencil.Region{N}, ::Tesserae.Stencil.Region{N}) where {N}
Base.split(::Tesserae.Stencil.Region{N}, ::Tesserae.Stencil.AxisPlane{N}) where {N}
```

## Stencil updates

`@stencil` applies views and broadcast to an update equation. For each
assignment, it updates only the positions for which all explicit `Region`
reads and writes exist. Positions outside that common range keep their
previous values. In this example the shifted source has no fourth sample.

```jldoctest
julia> using Tesserae.Stencil

julia> C = Region(Cell(), (4,)); (ex,) = unitoffsets(C);

julia> u = [1, 2, 3, 4]; out = zeros(Int, C);

julia> @stencil out[C] = u[C+ex];

julia> Tuple(out)
(2, 3, 4, 0)
```

An `@stencil begin ... end` block executes its equations in order, each with its
own common range. The macro sees the array references written in the equation.
For indexing inside a helper, supply a region or halo that covers the helper's
reads. An ordinary `@views @.` update uses the normal array bounds checks.

## Sparse assembly

Use [`create_sparse_matrix`](@ref) to allocate the pattern from row and column
regions and the allowed offsets from rows to columns. Write coefficients with
`@stencil_assemble`. This example adds the contribution of each interior face
with coefficient `a[I]` to the cells on either side.

```julia
C = Region(Cell(), (4,))
(ex,) = unitoffsets(C)
F = Region(Face(1), C)
a = fill(1.0, F)
A = create_sparse_matrix(C, C, (0ex, -ex, ex); ndofs=1)
I = interior(F)
L, R = I-ex/2, I+ex/2
@stencil_assemble begin
    A[L, L] += a[I]
    A[L, R] -= a[I]
    A[R, L] -= a[I]
    A[R, R] += a[I]
end
```

`A[L, R]` pairs corresponding samples, so each face contributes to one entry
in that selection. An entry outside the preallocated pattern raises an error.
With multiple degrees of freedom, the right-hand side returns a local matrix
(or a vector when one side has one degree of freedom). It is evaluated once
per selected point, then all its components are written to the matrix.
Use `fillzero!(A)` before reassembling with `+=`, and the
[implicit utilities](implicit_utils.md) to select the free degrees of freedom
and extract the system to solve.

```@docs
Tesserae.Stencil.@stencil
Tesserae.Stencil.@stencil_assemble
```
