########
# Grid #
########

const Grid{N, T, NT <: NamedTuple{<: Any, <: Tuple{AbstractMesh, Vararg{AbstractArray}}}, I} = StructArray{T, N, NT, I}

get_mesh(grid::Grid) = getx(grid)
spacing(grid::Grid) = spacing(get_mesh(grid))
spacing_inv(grid::Grid) = spacing_inv(get_mesh(grid))

##########
# SpGrid #
##########

const SpGrid{N, T, NT <: NamedTuple{<: Any, <: Tuple{CartesianMesh, SpArray, Vararg{SpArray}}}, I} = StructArray{T, N, NT, I}

function check_gridproperty(::Type{GridProp}, ::Type{Vec{dim, T}}) where {GridProp, dim, T}
    V = fieldtype(GridProp, 1)
    if V !== Vec{dim,T}
        error("generate_grid: the first property of grid must be `Vec{$dim, $T}` for given mesh, got $V")
    end
    if !(isbitstype(GridProp))
        error("generate_grid: the property type of grid must be `isbitstype` type")
    end
end

"""
    generate_grid(GridProp, mesh)

Generate a background grid where each element is of type `GridProp`.
The first field of `GridProp` is designated for `mesh`, and thus its type must match `eltype(mesh)`.
The resulting grid is a [`StructArray`](https://github.com/JuliaArrays/StructArrays.jl).

# Examples
```jldoctest
julia> struct GridProp{dim, T}
           x  :: Vec{dim, T}
           m  :: Float64
           mv :: Vec{dim, T}
           f  :: Vec{dim, T}
           v  :: Vec{dim, T}
           vⁿ :: Vec{dim, T}
       end

julia> grid = generate_grid(GridProp{2, Float64}, CartesianMesh(0.5, (0,3), (0,2)));

julia> grid[1]
GridProp{2, Float64}([0.0, 0.0], 0.0, [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])

julia> grid.x
7×5 CartesianMesh{2, Float64, Vector{Float64}}:
 [0.0, 0.0]  [0.0, 0.5]  [0.0, 1.0]  [0.0, 1.5]  [0.0, 2.0]
 [0.5, 0.0]  [0.5, 0.5]  [0.5, 1.0]  [0.5, 1.5]  [0.5, 2.0]
 [1.0, 0.0]  [1.0, 0.5]  [1.0, 1.0]  [1.0, 1.5]  [1.0, 2.0]
 [1.5, 0.0]  [1.5, 0.5]  [1.5, 1.0]  [1.5, 1.5]  [1.5, 2.0]
 [2.0, 0.0]  [2.0, 0.5]  [2.0, 1.0]  [2.0, 1.5]  [2.0, 2.0]
 [2.5, 0.0]  [2.5, 0.5]  [2.5, 1.0]  [2.5, 1.5]  [2.5, 2.0]
 [3.0, 0.0]  [3.0, 0.5]  [3.0, 1.0]  [3.0, 1.5]  [3.0, 2.0]

julia> grid.v
7×5 Matrix{Vec{2, Float64}}:
 [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]
 [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]
 [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]
 [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]
 [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]
 [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]
 [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]
```
"""
function generate_grid(::Type{GridProp}, mesh::AbstractMesh) where {GridProp}
    generate_grid(Array, GridProp, mesh)
end

function generate_grid(::Type{Array}, ::Type{GridProp}, mesh::AbstractMesh) where {GridProp}
    check_gridproperty(GridProp, eltype(mesh))
    arrays = map(T->Array{T}(undef, size(mesh)), Base.tail(fieldtypes(GridProp)))
    fillzero!(StructArray{GridProp}(tuple(mesh, arrays...)))
end

# SpArray is designed for Cartesian mesh
function generate_grid(::Type{SpArray}, ::Type{GridProp}, mesh::CartesianMesh) where {GridProp}
    check_gridproperty(GridProp, eltype(mesh))
    spinds = SpIndices(size(mesh))
    arrays = map(T->SpArray{T}(spinds), Base.tail(fieldtypes(GridProp)))
    StructArray{GridProp}(tuple(mesh, arrays...))
end

get_spinds(A::SpGrid) = get_spinds(getproperty(A, 2))

function update_block_sparsity!(A::SpGrid, blkspy)
    n = update_block_sparsity!(get_spinds(A), blkspy)
    StructArrays.foreachfield(a->resize_data!(a,n), A)
    A
end

@inline isactive(A::SpGrid, I...) = (@_propagate_inbounds_meta; isactive(get_spinds(A), I...))

Base.show(io::IO, mime::MIME"text/plain", x::SpGrid) = show(io, mime, ShowSpArray(x))
Base.show(io::IO, x::SpGrid) = show(io, ShowSpArray(x))
