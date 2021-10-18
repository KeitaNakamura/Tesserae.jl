"""
    Grid([::Type{NodeState}], [::ShapeFunction], axes::AbstractVector...)

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
struct Grid{dim, T, F <: Union{Nothing, ShapeFunction}, Node, State <: SpArray{Node, dim}} <: AbstractArray{Vec{dim, T}, dim}
    shapefunction::F
    coordinates::Coordinate{dim, NTuple{dim, T}, NTuple{dim, Vector{T}}}
    gridsteps::NTuple{dim, T}
    state::State
    coordinate_system::Symbol
end

Base.size(x::Grid) = map(length, gridaxes(x))
gridsteps(x::Grid) = x.gridsteps
gridsteps(x::Grid, i::Int) = (@_propagate_inbounds_meta; gridsteps(x)[i])
gridaxes(x::Grid) = coordinateaxes(x.coordinates)
gridaxes(x::Grid, i::Int) = (@_propagate_inbounds_meta; gridaxes(x)[i])
gridorigin(x::Grid) = Vec(map(first, gridaxes(x)))

checkshapefunction(::Grid{<: Any, <: Any, Nothing}) = throw(ArgumentError("`Grid` must include the information of shape function, see help `?Grid` for more details."))
checkshapefunction(::Grid{<: Any, <: Any, <: ShapeFunction}) = nothing

not_supported_coordinate_system(coordinate_system) =
    throw(ArgumentError("coordinate system `$(coordinate_system)` is not supported, use `:normal` in 1D and 3D, and `:plane_strain` or `:axisymmetric` in 2D."))
function Grid(::Type{Node}, shapefunction, coordinates::Coordinate{dim}; coordinate_system = nothing) where {Node, dim}
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
    Grid(shapefunction, Coordinate(Array.(axes)), map(step, axes), state, coordinate_system)
end

Grid(::Type{Node}, shapefunction, axes::Tuple{Vararg{AbstractVector}}; kwargs...) where {Node} = Grid(Node, shapefunction, Coordinate(axes); kwargs...)
Grid(::Type{Node}, axes::Tuple{Vararg{AbstractVector}}; kwargs...) where {Node} = Grid(Node, nothing, axes; kwargs...)
Grid(shapefunction, axes::Tuple{Vararg{AbstractVector}}; kwargs...) = Grid(Nothing, shapefunction, axes; kwargs...)
Grid(axes::Tuple{Vararg{AbstractVector}}; kwargs...) = Grid(Nothing, nothing, axes; kwargs...)

Grid(Node::Type, shapefunction, axes::AbstractVector...; kwargs...) = Grid(Node, shapefunction, axes; kwargs...)
Grid(Node::Type, axes::AbstractVector...; kwargs...) = Grid(Node, nothing, axes; kwargs...)
Grid(shapefunction, axes::AbstractVector...; kwargs...) = Grid(Nothing, shapefunction, axes; kwargs...)
Grid(axes::AbstractVector...; kwargs...) = Grid(Nothing, nothing, axes; kwargs...)

@inline function Base.getindex(grid::Grid{dim}, i::Vararg{Int, dim}) where {dim}
    @boundscheck checkbounds(grid, i...)
    @inbounds Vec(grid.coordinates[i...])
end

"""
    Poingr.neighboring_nodes(grid, x::Vec, h::Real)

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
@inline function neighboring_nodes(grid::Grid{dim}, x::Vec{dim}, h::Real) where {dim}
    dx = gridsteps(grid)
    xmin = gridorigin(grid)
    ξ = Tuple((x - xmin) ./ dx)
    all(@. 0 ≤ ξ ≤ $size(grid)-1) || return CartesianIndices(ntuple(d->1:0, Val(dim)))
    imin = @. unsafe_trunc(Int,  ceil(ξ - h)) + 1
    imax = @. imin + unsafe_trunc(Int, ceil(2h)) - 1
    inds = CartesianIndices(@. UnitRange(imin, imax))
    CartesianIndices(grid) ∩ inds
end
@inline function neighboring_nodes(grid::Grid, x::Vec)
    checkshapefunction(grid)
    neighboring_nodes(grid, x, support_length(grid.shapefunction))
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

function pointsinblock!(ptsinblk::AbstractArray{Vector{Int}, dim}, grid::Grid{dim}, xₚ::AbstractVector) where {dim}
    empty!.(ptsinblk)
    @inbounds for p in eachindex(xₚ)
        I = whichblock(grid, xₚ[p])
        I === nothing && continue
        push!(ptsinblk[I], p)
    end
    ptsinblk
end

function pointsinblock(grid::Grid, xₚ::AbstractVector)
    ptsinblk = Array{Vector{Int}}(undef, blocksize(grid))
    @inbounds @simd for i in eachindex(ptsinblk)
        ptsinblk[i] = Int[]
    end
    pointsinblock!(ptsinblk, grid, xₚ)
end

function sparsity_pattern(grid::Grid, xₚ::AbstractVector)
    h = active_length(grid.shapefunction)
    spat_threads = [fill(false, size(grid)) for _ in 1:Threads.nthreads()]
    @inbounds Threads.@threads for x in xₚ
        spat = spat_threads[Threads.threadid()]
        inds = neighboring_nodes(grid, x, h)
        spat[inds] .= true
    end
    broadcast(|, spat_threads...)
end

# this seems to be faster when using `reordering_pointstate!`
function sparsity_pattern(grid::Grid, xₚ::AbstractVector, ptsinblk::AbstractArray{Vector{Int}})
    h = active_length(grid.shapefunction)
    spat = falses(size(grid))
    for color in coloringblocks(size(grid))
        Threads.@threads for blockindex in color
            @inbounds for p in ptsinblk[blockindex]
                inds = neighboring_nodes(grid, xₚ[p], h)
                spat[inds] .= true
            end
        end
    end
    spat
end


struct BlockStepIndices{N} <: AbstractArray{CartesianIndex{N}, N}
    inds::Coordinate{N, NTuple{N, Int}, NTuple{N, StepRange{Int, Int}}}
end
Base.size(x::BlockStepIndices) = size(x.inds)
Base.getindex(x::BlockStepIndices{N}, i::Vararg{Int, N}) where {N} = (@_propagate_inbounds_meta; CartesianIndex(x.inds[i...]))

function coloringblocks(dims::NTuple{dim, Int}) where {dim}
    ncells = dims .- 1
    starts = SArray{NTuple{dim, 2}}(Iterators.ProductIterator(nfill((1,2), Val(dim)))...)
    nblocks = @. (ncells - 1) >> BLOCK_UNIT + 1
    vec(map(st -> BlockStepIndices(Coordinate(StepRange.(st, 2, nblocks))), starts))
end

struct Bound{dim, CI <: CartesianIndices{dim}}
    n::Vec{dim, Int}
    indices::CI
end

invalid_boundary_name_error(name) = error("invalid boundary name, got \"$name\", choose from \"-x\", \"+x\", \"-y\", \"+y\", \"-z\" and \"+z\"")
function Bound(grid::AbstractArray{<: Any, 1}, which::String)
    CI = CartesianIndices
    rng(x) = x:x
    which == "-x" && return Bound(Vec(-1), CI((rng(firstindex(grid, 1)),)))
    which == "+x" && return Bound(Vec( 1), CI((rng( lastindex(grid, 1)),)))
    invalid_boundary_name_error(name)
end
function Bound(grid::AbstractArray{<: Any, 2}, which::String)
    CI = CartesianIndices
    rng(x) = x:x
    which == "-x" && return Bound(Vec(-1, 0), CI((rng(firstindex(grid, 1)),          axes(grid, 2))))
    which == "+x" && return Bound(Vec( 1, 0), CI((rng( lastindex(grid, 1)),          axes(grid, 2))))
    which == "-y" && return Bound(Vec( 0,-1), CI((          axes(grid, 1), rng(firstindex(grid, 2)))))
    which == "+y" && return Bound(Vec( 0, 1), CI((          axes(grid, 1), rng( lastindex(grid, 2)))))
    invalid_boundary_name_error(name)
end
function Bound(grid::AbstractArray{<: Any, 3}, which::String)
    CI = CartesianIndices
    rng(x) = x:x
    which == "-x" && return Bound(Vec(-1, 0, 0), CI((rng(firstindex(grid, 1)),           axes(grid, 2),            axes(grid, 3))))
    which == "+x" && return Bound(Vec( 1, 0, 0), CI((rng( lastindex(grid, 1)),           axes(grid, 2),            axes(grid, 3))))
    which == "-y" && return Bound(Vec( 0,-1, 0), CI((          axes(grid, 1),  rng(firstindex(grid, 2)),           axes(grid, 3))))
    which == "+y" && return Bound(Vec( 0, 1, 0), CI((          axes(grid, 1),  rng( lastindex(grid, 2)),           axes(grid, 3))))
    which == "-z" && return Bound(Vec( 0, 0,-1), CI((          axes(grid, 1),            axes(grid, 2),  rng(firstindex(grid, 3)))))
    which == "+z" && return Bound(Vec( 0, 0, 1), CI((          axes(grid, 1),            axes(grid, 2),  rng( lastindex(grid, 3)))))
    invalid_boundary_name_error(name)
end

eachboundary(grid::AbstractArray{<: Any, 1}) = (Bound(grid, "-x"), Bound(grid, "+x"))
eachboundary(grid::AbstractArray{<: Any, 2}) = (Bound(grid, "-x"), Bound(grid, "+x"), Bound(grid, "-y"), Bound(grid, "+y"))
eachboundary(grid::AbstractArray{<: Any, 3}) = (Bound(grid, "-x"), Bound(grid, "+x"), Bound(grid, "-y"), Bound(grid, "+y"), Bound(grid, "-z"), Bound(grid, "+z"))
