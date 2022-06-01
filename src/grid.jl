"""
    Grid([::Type{NodeState}], [::Interpolation], axes::AbstractVector...)

Construct `Grid` by `axes`.

# Examples
```jldoctest
julia> Grid(range(0, 3, step = 1.0), range(1, 4, step = 1.0))
4×4 Grid{Float64, 2, Nothing, Nothing, Marble.SpArray{Nothing, 2, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}, PlaneStrain}:
 [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]
 [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]
 [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]
 [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]
```
"""
struct Grid{T, dim, F <: Union{Nothing, Interpolation}, Node, State <: SpArray{Node, dim}, CS <: CoordinateSystem} <: AbstractArray{Vec{dim, T}, dim}
    interpolation::F
    axes::NTuple{dim, Vector{T}}
    gridsteps::NTuple{dim, T}
    gridsteps_inv::NTuple{dim, T}
    state::State
    coordinate_system::CS
end

Base.size(x::Grid) = map(length, gridaxes(x))
gridsteps(x::Grid) = x.gridsteps
gridsteps(x::Grid, i::Int) = (@_propagate_inbounds_meta; gridsteps(x)[i])
gridsteps_inv(x::Grid) = x.gridsteps_inv
gridsteps_inv(x::Grid, i::Int) = (@_propagate_inbounds_meta; gridsteps_inv(x)[i])
gridaxes(x::Grid) = x.axes
gridaxes(x::Grid, i::Int) = (@_propagate_inbounds_meta; gridaxes(x)[i])
gridorigin(x::Grid) = Vec(map(first, gridaxes(x)))

check_interpolation(::Grid{<: Any, <: Any, Nothing}) = throw(ArgumentError("no interpolation information in `Grid`"))
check_interpolation(::Grid{<: Any, <: Any, <: Interpolation}) = nothing

# `axes` isa `Tuple`
function Grid{T}(::Type{Node}, interp, axes::NTuple{dim, AbstractVector}; coordinate_system = nothing) where {T, dim, Node}
    state = SpArray(StructVector{Node}(undef, 0), SpPattern(map(length, axes)))
    dx = map(step, axes)
    dx⁻¹ = map(inv, dx)

    Grid(
        interp,
        map(Array{T}, axes),
        map(T, dx),
        map(T, dx⁻¹),
        state,
        get_coordinate_system(coordinate_system, Val(dim)),
    )
end
function Grid{T}(interp::Interpolation, axes::NTuple{dim, AbstractVector}; kwargs...) where {T, dim}
    Node = default_nodestate_type(interp, Val(dim), Val(T))
    Grid{T}(Node, interp, axes; kwargs...)
end
Grid{T}(axes::Tuple{Vararg{AbstractVector}}; kwargs...) where {T} = Grid{T}(Nothing, nothing, axes; kwargs...)

# `axes` isa `Vararg`
Grid{T}(Node::Type, interp, axes::AbstractVector...; kwargs...) where {T} = Grid{T}(Node, interp, axes; kwargs...)
Grid{T}(interp::Interpolation, axes::AbstractVector...; kwargs...) where {T} = Grid{T}(interp, axes; kwargs...)
Grid{T}(axes::AbstractVector...; kwargs...) where {T} = Grid{T}(Nothing, nothing, axes; kwargs...)

# `T == Float64` by default
Grid(args...; kwargs...) = Grid{Float64}(args...; kwargs...)

@inline function Base.getindex(grid::Grid{<: Any, dim}, i::Vararg{Int, dim}) where {dim}
    @boundscheck checkbounds(grid, i...)
    @inbounds Vec(map(getindex, grid.axes, i))
end

"""
    Marble.neighbornodes(grid, x::Vec, h)

Return `CartesianIndices` storing neighboring node indices around `x`.
`h` is a range for searching and its unit is `gridsteps` `dx`.
In 1D, for example, the searching range becomes `x ± h*dx`.

# Examples
```jldoctest
julia> grid = Grid(0.0:1.0:5.0)
6-element Grid{Float64, 1, Nothing, Nothing, Marble.SpArray{Nothing, 1, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}, Marble.OneDimensional}:
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
6×6 Grid{Float64, 2, Nothing, Nothing, Marble.SpArray{Nothing, 2, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}, PlaneStrain}:
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
11×11 Grid{Float64, 2, Nothing, Nothing, Marble.SpArray{Nothing, 2, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}, PlaneStrain}:
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

blocksize(grid::Grid) = (ncells = size(grid) .- 1; @. (ncells - 1) >> BLOCK_UNIT + 1)

struct BlockStepIndices{N} <: AbstractArray{CartesianIndex{N}, N}
    inds::NTuple{N, StepRange{Int, Int}}
end
Base.size(x::BlockStepIndices) = map(length, x.inds)
Base.getindex(x::BlockStepIndices{N}, i::Vararg{Int, N}) where {N} = (@_propagate_inbounds_meta; CartesianIndex(map(getindex, x.inds, i)))

function threadsafe_blocks(gridsize::NTuple{dim, Int}) where {dim}
    ncells = gridsize .- 1
    starts = SArray{NTuple{dim, 2}}(Iterators.product(nfill((1,2), Val(dim))...)...)
    nblocks = @. (ncells - 1) >> BLOCK_UNIT + 1
    vec(map(st -> BlockStepIndices(StepRange.(st, 2, nblocks)), starts))
end


struct Boundaries{dim} <: AbstractArray{Tuple{CartesianIndex{dim}, Vec{dim, Int}}, dim}
    inds::CartesianIndices{dim}
    n::Vec{dim, Int}
end
Base.IndexStyle(::Type{<: Boundaries}) = IndexCartesian()
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

    inds = CartesianIndices(ntuple(d -> d==axis ? (index:index) : axes(grid, d), Val(dim)))
    n = Vec(ntuple(d -> ifelse(d==axis, dir, 0), Val(dim)))

    Boundaries(inds, n)
end
function boundaries(grid::AbstractArray, which::Vararg{String, N}) where {N}
    Iterators.flatten(ntuple(i -> _boundaries(grid, which[i]), Val(N)))
end
