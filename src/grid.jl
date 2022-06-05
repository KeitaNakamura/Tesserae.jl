struct AxisArray{dim, T, V <: Union{AbstractVector{T}, Tuple{Vararg{T}}}} <: AbstractArray{NTuple{dim, T}, dim}
    axes::NTuple{dim, V}
end
get_axes(A::AxisArray) = A.axes
Base.size(A::AxisArray) = map(length, A.axes)
@inline Base.getindex(A::AxisArray{dim}, i::Vararg{Int, dim}) where {dim} = (@_propagate_inbounds_meta; map(getindex, A.axes, i))

"""
    Grid(axes::AbstractVector...)
    Grid{T}(axes::AbstractVector...)

Construct `Grid` by `axes`.
`axes` must have `step` function, i.e., each axis should be linearly spaced.

# Examples
```jldoctest
julia> Grid(range(0, 3, step = 1.0), range(1, 4, step = 1.0))
4×4 Grid{Float64, 2, PlaneStrain}:
 [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]
 [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]
 [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]
 [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]
```
"""
struct Grid{T, dim, C <: CoordinateSystem} <: AbstractArray{Vec{dim, T}, dim}
    axisarray::AxisArray{dim, T, Vector{T}}
    gridsteps::NTuple{dim, T}
    gridsteps_inv::NTuple{dim, T}
    coordinate_system::C
end

get_axisarray(x::Grid) = x.axisarray
Base.size(x::Grid) = size(get_axisarray(x))
# grid helpers
gridsteps(x::Grid) = x.gridsteps
gridsteps(x::Grid, i::Int) = (@_propagate_inbounds_meta; gridsteps(x)[i])
gridsteps_inv(x::Grid) = x.gridsteps_inv
gridsteps_inv(x::Grid, i::Int) = (@_propagate_inbounds_meta; gridsteps_inv(x)[i])
gridaxes(x::Grid) = get_axes(x.axisarray)
gridaxes(x::Grid, i::Int) = (@_propagate_inbounds_meta; gridaxes(x)[i])
gridorigin(x::Grid) = Vec(map(first, gridaxes(x)))

function Grid{T}(axes::NTuple{dim, AbstractVector}; coordinate_system = nothing) where {T, dim}
    @assert all(map(issorted, axes))
    dx = map(step, axes)
    dx⁻¹ = map(inv, dx)
    Grid(
        AxisArray(map(Array{T}, axes)),
        map(T, dx),
        map(T, dx⁻¹),
        get_coordinate_system(coordinate_system, Val(dim)),
    )
end
Grid{T}(axes::AbstractVector...; kwargs...) where {T} = Grid{T}(axes; kwargs...)
Grid(args...; kwargs...) = Grid{Float64}(args...; kwargs...)

@inline function Base.getindex(grid::Grid{<: Any, dim}, i::Vararg{Int, dim}) where {dim}
    @boundscheck checkbounds(grid, i...)
    @inbounds Vec(get_axisarray(grid)[i...])
end

isinside(x::Vec, grid::Grid) = all(first(grid) .≤ x .≤ last(grid))

"""
    Marble.neighbornodes(grid, x::Vec, h)

Return `CartesianIndices` storing neighboring node indices around `x`.
`h` is a range for searching and its unit is `gridsteps` `dx`.
In 1D, for example, the searching range becomes `x ± h*dx`.

# Examples
```jldoctest
julia> grid = Grid(0.0:1.0:5.0)
6-element Grid{Float64, 1, Marble.OneDimensional}:
 [0.0]
 [1.0]
 [2.0]
 [3.0]
 [4.0]
 [5.0]

julia> Marble.neighbornodes(grid, Vec(1.5), 1)
2-element CartesianIndices{1, Tuple{UnitRange{Int64}}}:
 CartesianIndex(2,)
 CartesianIndex(3,)

julia> Marble.neighbornodes(grid, Vec(1.5), 2)
4-element CartesianIndices{1, Tuple{UnitRange{Int64}}}:
 CartesianIndex(1,)
 CartesianIndex(2,)
 CartesianIndex(3,)
 CartesianIndex(4,)
```
"""
@inline function neighbornodes(grid::Grid{<: Any, dim}, x::Vec{dim}, h) where {dim}
    dx⁻¹ = gridsteps_inv(grid)
    xmin = gridorigin(grid)
    ξ = Tuple((x - xmin) .* dx⁻¹)
    T = eltype(ξ)
    all(@. zero(T) ≤ ξ ≤ T($size(grid)-1)) || return CartesianIndices(nfill(1:0, Val(dim)))
    # To handle zero division in nodal calculations such as fᵢ/mᵢ, we use a bit small `h`.
    # This means `neighbornodes` doesn't include bounds of range.
    _neighbornodes(size(grid), ξ, @. T(h) - sqrt(eps(T)))
end
@inline function _neighbornodes(dims::Dims, ξ, h)
    imin = Tuple(@. max(unsafe_trunc(Int,  ceil(ξ - h)) + 1, 1))
    imax = Tuple(@. min(unsafe_trunc(Int, floor(ξ + h)) + 1, dims))
    CartesianIndices(@. UnitRange(imin, imax))
end

"""
    Marble.whichcell(grid, x::Vec)

Return cell index where `x` locates.

# Examples
```jldoctest
julia> grid = Grid(0.0:1.0:5.0, 0.0:1.0:5.0)
6×6 Grid{Float64, 2, PlaneStrain}:
 [0.0, 0.0]  [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]  [0.0, 5.0]
 [1.0, 0.0]  [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]  [1.0, 5.0]
 [2.0, 0.0]  [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]  [2.0, 5.0]
 [3.0, 0.0]  [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]  [3.0, 5.0]
 [4.0, 0.0]  [4.0, 1.0]  [4.0, 2.0]  [4.0, 3.0]  [4.0, 4.0]  [4.0, 5.0]
 [5.0, 0.0]  [5.0, 1.0]  [5.0, 2.0]  [5.0, 3.0]  [5.0, 4.0]  [5.0, 5.0]

julia> Marble.whichcell(grid, Vec(1.5, 1.5))
CartesianIndex(2, 2)
```
"""
@inline function whichcell(grid::Grid{<: Any, dim}, x::Vec{dim}) where {dim}
    dx⁻¹ = gridsteps_inv(grid)
    xmin = gridorigin(grid)
    ξ = Tuple((x - xmin) .* dx⁻¹)
    all(@. 0 ≤ ξ ≤ $size(grid)-1) || return nothing
    CartesianIndex(@. unsafe_trunc(Int, floor(ξ)) + 1)
end

"""
    Marble.whichblock(grid, x::Vec)

Return block index where `x` locates.
The unit block size is `2^$BLOCK_UNIT` cells.

# Examples
```jldoctest
julia> grid = Grid(0.0:1.0:10.0, 0.0:1.0:10.0)
11×11 Grid{Float64, 2, PlaneStrain}:
 [0.0, 0.0]   [0.0, 1.0]   [0.0, 2.0]   …  [0.0, 9.0]   [0.0, 10.0]
 [1.0, 0.0]   [1.0, 1.0]   [1.0, 2.0]      [1.0, 9.0]   [1.0, 10.0]
 [2.0, 0.0]   [2.0, 1.0]   [2.0, 2.0]      [2.0, 9.0]   [2.0, 10.0]
 [3.0, 0.0]   [3.0, 1.0]   [3.0, 2.0]      [3.0, 9.0]   [3.0, 10.0]
 [4.0, 0.0]   [4.0, 1.0]   [4.0, 2.0]      [4.0, 9.0]   [4.0, 10.0]
 [5.0, 0.0]   [5.0, 1.0]   [5.0, 2.0]   …  [5.0, 9.0]   [5.0, 10.0]
 [6.0, 0.0]   [6.0, 1.0]   [6.0, 2.0]      [6.0, 9.0]   [6.0, 10.0]
 [7.0, 0.0]   [7.0, 1.0]   [7.0, 2.0]      [7.0, 9.0]   [7.0, 10.0]
 [8.0, 0.0]   [8.0, 1.0]   [8.0, 2.0]      [8.0, 9.0]   [8.0, 10.0]
 [9.0, 0.0]   [9.0, 1.0]   [9.0, 2.0]      [9.0, 9.0]   [9.0, 10.0]
 [10.0, 0.0]  [10.0, 1.0]  [10.0, 2.0]  …  [10.0, 9.0]  [10.0, 10.0]

julia> Marble.whichblock(grid, Vec(8.5, 1.5))
CartesianIndex(2, 1)
```
"""
@inline function whichblock(grid::Grid, x::Vec)
    I = whichcell(grid, x)
    I === nothing && return nothing
    CartesianIndex(@. ($Tuple(I)-1) >> BLOCK_UNIT + 1)
end

blocksize(gridsize::Tuple{Vararg{Int}}) = (ncells = gridsize .- 1; @. (ncells - 1) >> BLOCK_UNIT + 1)

function threadsafe_blocks(blocksize::NTuple{dim, Int}) where {dim}
    starts = AxisArray(nfill((1,2), Val(dim)))
    tuple2cartesian(x) = LazyDotArray(CartesianIndex{dim}, x)
    vec(map(st -> tuple2cartesian(AxisArray(StepRange.(st, 2, blocksize))), starts))
end


struct Boundaries{dim} <: AbstractArray{Tuple{CartesianIndex{dim}, Vec{dim, Int}}, dim}
    inds::CartesianIndices{dim, NTuple{dim, UnitRange{Int}}}
    n::Vec{dim, Int}
end
Base.size(x::Boundaries) = size(x.inds)
Base.getindex(x::Boundaries{dim}, I::Vararg{Int, dim}) where {dim} = (@_propagate_inbounds_meta; (x.inds[I...], x.n))

function _boundaries(grid::AbstractArray{<: Any, dim}, which::String) where {dim}
    if     which[2] == 'x'; axis = 1
    elseif which[2] == 'y'; axis = 2
    elseif which[2] == 'z'; axis = 3
    else error("invalid bound name")
    end

    if     which[1] == '-'; index = firstindex(grid, axis); dir =  1
    elseif which[1] == '+'; index =  lastindex(grid, axis); dir = -1
    else error("invalid bound name")
    end

    rngs = ntuple(d -> d==axis ? (index:index) : UnitRange(axes(grid, d)), Val(dim))
    n = Vec{dim}(i-> ifelse(i==axis, dir, 0))

    Boundaries(CartesianIndices(rngs), n)
end
function gridbounds(grid::AbstractArray, which::String...)
    Iterators.flatten(map_tuple(_boundaries, grid, which))
end
