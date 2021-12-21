struct NodePosition
    nth::Int
    dir::Int # 1 or -1
end
nthfrombound(pos::NodePosition) = pos.nth
dirfrombound(pos::NodePosition) = pos.dir

struct BoundaryCondition{dim}
    boundcontours::NTuple{dim, Array{Int, dim}}
    node_positions::NTuple{dim, Array{NodePosition, dim}}
end

node_position(bc::BoundaryCondition, I, d) = (@_propagate_inbounds_meta; bc.node_positions[d][I])
node_position(bc::BoundaryCondition{dim}, I) where {dim} = (@_propagate_inbounds_meta; ntuple(d -> node_position(bc, I, d), Val(dim)))

@inline function isonbound(bc::BoundaryCondition, I, d::Int)
    @_propagate_inbounds_meta
    pos = node_position(bc, I, d)
    pos.nth === 0 && pos.dir !== 0
end
@inline function isonbound(bc::BoundaryCondition{dim}, I) where {dim}
    @_propagate_inbounds_meta
    |(ntuple(d -> isonbound(bc, I, d), Val(dim))...)
end

@inline function isinbound(bc::BoundaryCondition, I, d::Int)
    @_propagate_inbounds_meta
    pos = node_position(bc, I, d)
    pos.nth === 0 && pos.dir === 0
end
@inline function isinbound(bc::BoundaryCondition{dim}, I) where {dim}
    @_propagate_inbounds_meta
    prod(ntuple(d -> isinbound(bc, I, d), Val(dim)))
end

function set_boundcontour!(boundcontour::AbstractVector{Int}, start::Int, dir::Int)
    @boundscheck checkbounds(boundcontour, start)
    @assert dir == 1 || dir == -1
    nth = 0
    for i in start:dir:ifelse(dir==1, lastindex(boundcontour), firstindex(boundcontour))
        if nth < boundcontour[i]
            boundcontour[i] = nth
        end
        nth += 1
    end
    boundcontour
end

function set_boundcontour!(boundcontour::AbstractVector{Int}, withinbounds::AbstractVector{Bool})
    @assert size(boundcontour) == size(withinbounds)
    set_boundcontour!(boundcontour, firstindex(boundcontour), 1)
    set_boundcontour!(boundcontour, lastindex(boundcontour), -1)
    for i in eachindex(withinbounds)
        if withinbounds[i] == true
            set_boundcontour!(boundcontour, i,  1)
            set_boundcontour!(boundcontour, i, -1)
        end
    end
    boundcontour
end

function eachaxis(x::AbstractArray{<: Any, dim}, d::Int) where {dim}
    @assert d ≤ ndims(x)
    _colon(x) = x:x
    _eachindex(x, i) = 1:size(x, i)
    axisindices(x, I) = ntuple(i -> i == d ? _eachindex(x, i) : _colon(I[i]), Val(dim))
    slice = CartesianIndices(ntuple(i -> i == d ? _colon(1) : _eachindex(x, i), Val(dim)))
    (vec(view(x, axisindices(x, I)...)) for I in slice)
end

function _direction(contour::AbstractVector{Int})
    @inbounds begin
        if length(contour) == 2
            contour[1] < contour[2] && return 1
            contour[1] > contour[2] && return -1
            return 0
        elseif length(contour) == 3
            (contour[1] < contour[2] || contour[2] < contour[3]) && return 1
            (contour[1] > contour[2] || contour[2] > contour[3]) && return -1
            return 0
        else
            error("unreachable")
        end
    end
end

function BoundaryCondition(withinbounds::AbstractArray{Bool, dim}) where {dim}
    boundcontours = ntuple(d -> Array{Int}(undef, size(withinbounds)), Val(dim))
    node_positions = ntuple(d -> Array{NodePosition}(undef, size(withinbounds)), Val(dim))
    bc = BoundaryCondition(boundcontours, node_positions)
    setbounds!(bc, withinbounds)
    bc
end

function setbounds!(bc::BoundaryCondition{dim}, withinbounds::AbstractArray{Bool, dim}) where {dim}
    for d in 1:dim
        # update boundcontours
        boundcontour = bc.boundcontours[d]
        @assert size(boundcontour) == size(withinbounds)
        fill!(boundcontour, size(boundcontour, d) + 1) # set large value for initialization
        for args in zip(eachaxis(boundcontour, d), eachaxis(withinbounds, d))
            set_boundcontour!(args...)
        end
        # update node_positions
        node_position = bc.node_positions[d]
        for (axis, contour) in zip(eachaxis(node_position, d), eachaxis(boundcontour, d))
            @assert length(axis) == length(contour)
            @inbounds for i in eachindex(axis)
                inds = intersect(i-1:i+1, eachindex(axis))
                axis[i] = NodePosition(contour[i], _direction(@view contour[inds]))
            end
        end
    end
end


"""
    Grid([::Type{NodeState}], [::Interpolation], axes::AbstractVector...)

Construct `Grid` by `axes`.

# Examples
```jldoctest
julia> Grid(range(0, 3, step = 1.0), range(1, 4, step = 1.0))
4×4 Grid{2, Float64, Nothing, Nothing, Poingr.SpArray{Nothing, 2, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}}:
 [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]
 [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]
 [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]
 [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]
```
"""
struct Grid{dim, T, F <: Union{Nothing, Interpolation}, Node, State <: SpArray{Node, dim}} <: AbstractArray{Vec{dim, T}, dim}
    interpolation::F
    coordinates::Coordinate{dim, NTuple{dim, T}, NTuple{dim, Vector{T}}}
    gridsteps::NTuple{dim, T}
    state::State
    coordinate_system::Symbol
    bc::BoundaryCondition{dim}
end

Base.size(x::Grid) = map(length, gridaxes(x))
gridsteps(x::Grid) = x.gridsteps
gridsteps(x::Grid, i::Int) = (@_propagate_inbounds_meta; gridsteps(x)[i])
gridaxes(x::Grid) = coordinateaxes(x.coordinates)
gridaxes(x::Grid, i::Int) = (@_propagate_inbounds_meta; gridaxes(x)[i])
gridorigin(x::Grid) = Vec(map(first, gridaxes(x)))

node_position(grid::Grid, I) = (@_propagate_inbounds_meta; node_position(grid.bc, I))
node_position(grid::Grid, I, d) = (@_propagate_inbounds_meta; node_position(grid.bc, I, d))
isonbound(grid::Grid, I) = (@_propagate_inbounds_meta; isonbound(grid.bc, I))
isonbound(grid::Grid, I, d::Int) = (@_propagate_inbounds_meta; isonbound(grid.bc, I, d))
isinbound(grid::Grid, I) = (@_propagate_inbounds_meta; isinbound(grid.bc, I))
isinbound(grid::Grid, I, d::Int) = (@_propagate_inbounds_meta; isinbound(grid.bc, I, d))
setbounds!(grid::Grid, withinbounds::AbstractArray{Bool}) = setbounds!(grid.bc, withinbounds)

check_interpolation(::Grid{<: Any, <: Any, Nothing}) = throw(ArgumentError("`Grid` must include the information of interpolation, see help `?Grid` for more details."))
check_interpolation(::Grid{<: Any, <: Any, <: Interpolation}) = nothing

not_supported_coordinate_system(coordinate_system) =
    throw(ArgumentError("coordinate system `$(coordinate_system)` is not supported, use `:normal` in 1D and 3D, and `:plane_strain` or `:axisymmetric` in 2D."))
function Grid(::Type{Node}, interp, coordinates::Coordinate{dim}; coordinate_system = nothing, withinbounds = falses(size(coordinates))) where {Node, dim}
    state = SpArray(StructVector{Node}(undef, 0), SpPattern(size(coordinates)))
    axes = coordinateaxes(coordinates)
    if coordinate_system !== nothing
        coordinate_system = Symbol(coordinate_system) # handle string
        if dim == 2
            if coordinate_system != :plane_strain && coordinate_system != :axisymmetric
                not_supported_coordinate_system(coordinate_system)
            end
        else
            if coordinate_system != :normal
                not_supported_coordinate_system(coordinate_system)
            end
        end
    else
        if dim == 2
            coordinate_system = :plane_strain
        else
            coordinate_system = :normal
        end
    end
    Grid(interp, Coordinate(Array.(axes)), map(step, axes), state, coordinate_system, BoundaryCondition(withinbounds))
end

function Grid(interp::Interpolation, coordinates::Coordinate{dim, Tup}; kwargs...) where {dim, Tup}
    T = promote_type(Tup.parameters...)
    Node = default_nodestate_type(interp, Val(dim), Val(T))
    Grid(Node, interp, coordinates; kwargs...)
end

# `interp` must be given if Node is given
Grid(::Type{Node}, interp, axes::Tuple{Vararg{AbstractVector}}; kwargs...) where {Node} = Grid(Node, interp, Coordinate(axes); kwargs...)
Grid(interp::Interpolation, axes::Tuple{Vararg{AbstractVector}}; kwargs...) = Grid(interp, Coordinate(axes); kwargs...)
Grid(axes::Tuple{Vararg{AbstractVector}}; kwargs...) = Grid(Nothing, nothing, axes; kwargs...)

# `interp` must be given if Node is given
Grid(Node::Type, interp, axes::AbstractVector...; kwargs...) = Grid(Node, interp, axes; kwargs...)
Grid(interp::Interpolation, axes::AbstractVector...; kwargs...) = Grid(interp, axes; kwargs...)
Grid(axes::AbstractVector...; kwargs...) = Grid(Nothing, nothing, axes; kwargs...)

@inline function Base.getindex(grid::Grid{dim}, i::Vararg{Int, dim}) where {dim}
    @boundscheck checkbounds(grid, i...)
    @inbounds Vec(grid.coordinates[i...])
end

"""
    Poingr.neighboring_nodes(grid, x::Vec, h)

Return `CartesianIndices` storing neighboring node indices around `x`.
`h` is a range for searching and its unit is `gridsteps` `dx`.
In 1D, for example, the searching range becomes `x ± h*dx`.

# Examples
```jldoctest
julia> grid = Grid(0.0:1.0:5.0)
6-element Grid{1, Float64, Nothing, Nothing, Poingr.SpArray{Nothing, 1, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}}:
 [0.0]
 [1.0]
 [2.0]
 [3.0]
 [4.0]
 [5.0]

julia> Poingr.neighboring_nodes(grid, Vec(1.5), 1)
2-element CartesianIndices{1, Tuple{UnitRange{Int64}}}:
 CartesianIndex(2,)
 CartesianIndex(3,)

julia> Poingr.neighboring_nodes(grid, Vec(1.5), 2)
4-element CartesianIndices{1, Tuple{UnitRange{Int64}}}:
 CartesianIndex(1,)
 CartesianIndex(2,)
 CartesianIndex(3,)
 CartesianIndex(4,)
```
"""
@inline function neighboring_nodes(grid::Grid{dim}, x::Vec{dim}, h) where {dim}
    dx = gridsteps(grid)
    xmin = gridorigin(grid)
    ξ = Tuple((x - xmin) ./ dx)
    T = eltype(ξ)
    all(@. zero(T) ≤ ξ ≤ T($size(grid)-1)) || return CartesianIndices(nfill(1:0, Val(dim)))
    # To handle zero division in nodal calculations such as fᵢ/mᵢ, we use a bit small `h`.
    # This means `neighboring_nodes` doesn't include bounds of range.
    _neighboring_nodes(grid, ξ, h .- sqrt(eps(T)))
end
@inline function neighboring_nodes(grid::Grid, x::Vec)
    check_interpolation(grid)
    neighboring_nodes(grid, x, support_length(grid.interpolation))
end

@inline function _neighboring_nodes(grid::Grid, ξ, h)
    imin = Tuple(@. unsafe_trunc(Int, ceil(ξ - h))  + 1)
    imax = Tuple(@. unsafe_trunc(Int, floor(ξ + h)) + 1)
    inds = CartesianIndices(@. UnitRange(imin, imax))
    CartesianIndices(grid) ∩ inds
end


"""
    Poingr.neighboring_cells(grid, x::Vec, h::Int)
    Poingr.neighboring_cells(grid, cellindex::CartesianIndex, h::Int)

Return `CartesianIndices` storing neighboring cell indices around `x`.
`h` is number of outer cells around cell where `x` locates.
In 1D, for example, the searching range becomes `x ± h*dx`.

# Examples
```jldoctest
julia> grid = Grid(0.0:1.0:5.0, 0.0:1.0:5.0)
6×6 Grid{2, Float64, Nothing, Nothing, Poingr.SpArray{Nothing, 2, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}}:
 [0.0, 0.0]  [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]  [0.0, 5.0]
 [1.0, 0.0]  [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]  [1.0, 5.0]
 [2.0, 0.0]  [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]  [2.0, 5.0]
 [3.0, 0.0]  [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]  [3.0, 5.0]
 [4.0, 0.0]  [4.0, 1.0]  [4.0, 2.0]  [4.0, 3.0]  [4.0, 4.0]  [4.0, 5.0]
 [5.0, 0.0]  [5.0, 1.0]  [5.0, 2.0]  [5.0, 3.0]  [5.0, 4.0]  [5.0, 5.0]

julia> x = Vec(1.5, 1.5);

julia> Poingr.neighboring_cells(grid, x, 1)
3×3 CartesianIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}}:
 CartesianIndex(1, 1)  CartesianIndex(1, 2)  CartesianIndex(1, 3)
 CartesianIndex(2, 1)  CartesianIndex(2, 2)  CartesianIndex(2, 3)
 CartesianIndex(3, 1)  CartesianIndex(3, 2)  CartesianIndex(3, 3)

julia> Poingr.neighboring_cells(grid, Poingr.whichcell(grid, x), 1) == ans
true
```
"""
function neighboring_cells(grid::Grid{dim}, cellindex::CartesianIndex{dim}, h::Int) where {dim}
    inds = CartesianIndices(size(grid) .- 1)
    @boundscheck checkbounds(inds, cellindex)
    u = oneunit(cellindex)
    inds ∩ (cellindex-h*u:cellindex+h*u)
end

@inline function neighboring_cells(grid::Grid{dim}, x::Vec{dim}, h::Int) where {dim}
    neighboring_cells(grid, whichcell(grid, x), h)
end

function neighboring_blocks(grid::Grid{dim}, blockindex::CartesianIndex{dim}, h::Int) where {dim}
    inds = CartesianIndices(blocksize(grid))
    @boundscheck checkbounds(inds, blockindex)
    u = oneunit(blockindex)
    inds ∩ (blockindex-h*u:blockindex+h*u)
end

@inline function neighboring_blocks(grid::Grid{dim}, x::Vec{dim}, h::Int) where {dim}
    neighboring_blocks(grid, whichblock(grid, x), h)
end

"""
    Poingr.whichcell(grid, x::Vec)

Return cell index where `x` locates.

# Examples
```jldoctest
julia> grid = Grid(0.0:1.0:5.0, 0.0:1.0:5.0)
6×6 Grid{2, Float64, Nothing, Nothing, Poingr.SpArray{Nothing, 2, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}}:
 [0.0, 0.0]  [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]  [0.0, 5.0]
 [1.0, 0.0]  [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]  [1.0, 5.0]
 [2.0, 0.0]  [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]  [2.0, 5.0]
 [3.0, 0.0]  [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]  [3.0, 5.0]
 [4.0, 0.0]  [4.0, 1.0]  [4.0, 2.0]  [4.0, 3.0]  [4.0, 4.0]  [4.0, 5.0]
 [5.0, 0.0]  [5.0, 1.0]  [5.0, 2.0]  [5.0, 3.0]  [5.0, 4.0]  [5.0, 5.0]

julia> Poingr.whichcell(grid, Vec(1.5, 1.5))
CartesianIndex(2, 2)
```
"""
@inline function whichcell(grid::Grid{dim}, x::Vec{dim}) where {dim}
    dx = gridsteps(grid)
    xmin = gridorigin(grid)
    ξ = Tuple((x - xmin) ./ dx)
    all(@. 0 ≤ ξ ≤ $size(grid)-1) || return nothing
    CartesianIndex(@. unsafe_trunc(Int, floor(ξ)) + 1)
end

"""
    Poingr.whichblock(grid, x::Vec)

Return block index where `x` locates.
The unit block size is `2^$BLOCK_UNIT` cells.

# Examples
```jldoctest
julia> grid = Grid(0.0:1.0:10.0, 0.0:1.0:10.0)
11×11 Grid{2, Float64, Nothing, Nothing, Poingr.SpArray{Nothing, 2, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}}:
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

julia> Poingr.whichblock(grid, Vec(8.5, 1.5))
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
    inds::Coordinate{N, NTuple{N, Int}, NTuple{N, StepRange{Int, Int}}}
end
Base.size(x::BlockStepIndices) = size(x.inds)
Base.getindex(x::BlockStepIndices{N}, i::Vararg{Int, N}) where {N} = (@_propagate_inbounds_meta; CartesianIndex(x.inds[i...]))

function threadsafe_blocks(dims::NTuple{dim, Int}) where {dim}
    ncells = dims .- 1
    starts = SArray{NTuple{dim, 2}}(Coordinate(nfill((1,2), Val(dim)))...)
    nblocks = @. (ncells - 1) >> BLOCK_UNIT + 1
    vec(map(st -> BlockStepIndices(Coordinate(StepRange.(st, 2, nblocks))), starts))
end


struct Boundary{dim}
    n::Vec{dim, Int}
    I::CartesianIndex{dim}
end

function eachboundary(grid::Grid{dim, T}) where {dim, T}
    _dir(pos::NodePosition, d::Int) = Vec{dim}(i -> ifelse(i === d, -pos.dir, 0))
    function getbound(grid, I, d)
        @inbounds begin
            pos = node_position(grid, I, d)
            Boundary(_dir(pos, d), I)
        end
    end
    ntuple(Val(dim)) do d
        (getbound(grid, I, d) for I in CartesianIndices(grid) if @inbounds(isonbound(grid, I, d)))
    end |> Iterators.flatten
end
