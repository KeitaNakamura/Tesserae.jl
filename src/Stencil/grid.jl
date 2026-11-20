# Physical indices remain 1:n. Extra storage has logical halo indices 0, n+1, ... .
function storage_axes(r::Region{N}, halo::Integer) where {N}
    iswhole(r) || throw(ArgumentError("field allocation requires a whole-domain region"))
    width = Int(halo)
    width ≥ 0 || throw(ArgumentError("halo width must be nonnegative"))
    ntuple(d -> (1-width):(size(r, d)+width), Val(N))
end

function field_storage(A, axes::Tuple, halo::Integer)
    iszero(halo) ? A : OffsetArray(A, axes...)
end

"""
    zeros([T=Float64,] r::Region; halo=0)

Allocate zero-filled CPU storage at the whole-domain region `r`.
A positive `halo` wraps storage in an `OffsetArray`, retaining physical indices
`1:n` and adding equally many samples on both sides of each axis.
"""
function Base.zeros(::Type{T}, r::Region; halo::Integer=0) where {T}
    axes = storage_axes(r, halo)
    field_storage(zeros(T, map(length, axes)), axes, halo)
end
Base.zeros(r::Region; halo::Integer=0) = zeros(Float64, r;halo)

"""
    fill(value, r::Region; halo=0)

Allocate CPU storage filled with `value`, using the same placement and halo
conventions as `zeros(T, r; halo)`.
"""
function Base.fill(value, r::Region; halo::Integer=0)
    axes = storage_axes(r, halo)
    field_storage(fill(value, map(length, axes)), axes, halo)
end

"""
    similar(A, [T=eltype(A),] r::Region; halo=0)

Allocate uninitialized storage on `A`'s backend at the whole-domain region `r`.
The region and halo determine its axes, independently of `A`'s shape.
"""
function Base.similar(A::AbstractArray, ::Type{T}, r::Region; halo::Integer=0) where {T}
    axes = storage_axes(r, halo)
    field_storage(similar(no_offset_view(A), T, map(length, axes)), axes, halo)
end
Base.similar(A::AbstractArray, r::Region; halo::Integer=0) = similar(A, eltype(A), r;halo)

# The coordinate component retains placement and physical-domain information.
# Numerical properties remain ordinary StructArray components.
struct StaggeredMesh{N, T, A<:AbstractArray{Vec{N, T}, N}, R<:Region{N}} <: AbstractMesh{N, T, N}
    points::A
    region::R
end

Base.size(mesh::StaggeredMesh) = size(mesh.points)
Base.axes(mesh::StaggeredMesh) = axes(mesh.points)
Base.IndexStyle(::Type{<:StaggeredMesh}) = IndexCartesian()
@inline function Base.getindex(mesh::StaggeredMesh{N}, I::Vararg{Int, N}) where {N}
    @_propagate_inbounds_meta
    mesh.points[I...]
end

Base.copy(mesh::StaggeredMesh) = StaggeredMesh(copy(mesh.points), mesh.region)
spacing(mesh::StaggeredMesh) = spacing(no_offset_view(mesh.points))
spacing_inv(mesh::StaggeredMesh) = spacing_inv(no_offset_view(mesh.points))
volume(mesh::StaggeredMesh{N}) where {N} = prod(mesh.region.cells)*spacing(mesh)^N

Region(mesh::StaggeredMesh) = mesh.region

"""
    Region(grid)

Return the physical sampling region of a grid created with
`generate_grid(location, GridProp, mesh; halo)`. Its placement and cell counts
come from the coordinate component, independently of halo storage. For example,
`grid.p[Region(grid)]` selects only physical pressure samples.
"""
function Region(grid::Grid)
    mesh = get_mesh(grid)
    mesh isa StaggeredMesh || throw(ArgumentError("Region(grid) requires a grid created with an explicit sampling placement"))
    mesh.region
end

function Adapt.adapt_structure(to::AbstractDevice, mesh::StaggeredMesh)
    StaggeredMesh(Adapt.adapt(to, mesh.points), mesh.region)
end
# Region geometry is host metadata. A numerical kernel needs only the points.
Adapt.adapt_structure(to, mesh::StaggeredMesh) = Adapt.adapt(to, mesh.points)
KernelAbstractions.get_backend(mesh::StaggeredMesh) = KernelAbstractions.get_backend(no_offset_view(mesh.points))

"""
    generate_grid(location, GridProp, mesh::CartesianMesh; halo=0)
    generate_grid(Array, location, GridProp, mesh::CartesianMesh; halo=0)

Generate a StructArray at cell, face, edge or vertex sampling points on the
physical domain described by `mesh`. Declare properties as for MPM grids:
the first property stores coordinates and must match `eltype(mesh)`; the
remaining properties are initialized to zero. All components share the same
axes, including `halo` samples beyond each physical boundary.

Physical indices remain `1:n` at each placement; positive halos use logical
indices `0`, `n+1`, etc. `Region(grid)` returns the physical region, whereas
`size(grid)` and `axes(grid)` include halo storage. The input mesh bounds
specify the physical domain; use `halo` here to add neighbor storage.
Boundary values and every discrete numerical equation remain caller-defined.

```julia
CellProp = @NamedTuple{x::Vec{2, Float64}, p::Float64}
FaceProp = @NamedTuple{x::Vec{2, Float64}, v::Float64}
mesh = CartesianMesh(0.1, (0, 1), (0, 1))
cells = generate_grid(Cell(), CellProp, mesh; halo=1)
faces = [generate_grid(Face(d), FaceProp, mesh; halo=1) for d in 1:2]
C = Region(cells)
F = Region(faces[1])
```
"""
@inline function generate_grid(location::Location, ::Type{GridProp}, mesh::CartesianMesh; halo::Integer=0) where {GridProp}
    generate_grid(Array, location, GridProp, mesh;halo)
end

Base.@constprop :aggressive function generate_grid(::Type{Array}, location::Location, ::Type{GridProp}, mesh::CartesianMesh{N}; halo::Integer=0) where {GridProp, N}
    check_gridproperty(GridProp, eltype(mesh))
    location = canonical(location, N)
    region = Region(location, mesh)
    ax = storage_axes(region, halo)
    h = spacing(mesh)
    coordinate_axes = ntuple(Val(N)) do d
        collect(coordinate_axis(ax[d], isnodealigned(location, d), region.frame, d))
    end
    points = CartesianMesh(coordinate_axes, h, spacing_inv(mesh);block_size_log2=Val(block_size_log2(mesh)))
    dims = map(length, ax)
    coordinates = StaggeredMesh(field_storage(points, ax, halo), region)
    arrays = map(T -> field_storage(zeros(T, dims), ax, halo), Base.tail(fieldtypes(GridProp)))
    StructArray{GridProp}((coordinates, arrays...))
end
