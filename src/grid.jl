"""
    Grid([::Type{NodeState}], [::ShapeFunction], axes::AbstractVector...)

Construct `Grid` by `axes`.

# Examples
```jldoctest
julia> Grid(range(0, 3, step = 1.0), range(1, 4, step = 1.0))
4×4 Grid{2, Float64, Nothing, Nothing, Poingr.SpArray{Nothing, 2, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}, Tuple{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}}}:
 [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]
 [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]
 [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]
 [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]
```
"""
struct Grid{dim, T, F <: Union{Nothing, ShapeFunction{dim}}, Node, State <: SpArray{Node, dim}} <: AbstractArray{Vec{dim, T}, dim}
    shapefunction::F
    coordinates::Coordinate{dim, NTuple{dim, T}, NTuple{dim, Vector{T}}}
    gridsteps::NTuple{dim, T}
    state::State
end

Base.size(x::Grid) = map(length, gridaxes(x))
gridsteps(x::Grid) = x.gridsteps
gridsteps(x::Grid, i::Int) = (@_propagate_inbounds_meta; gridsteps(x)[i])
gridaxes(x::Grid) = coordinateaxes(x.coordinates)
gridaxes(x::Grid, i::Int) = (@_propagate_inbounds_meta; gridaxes(x)[i])
gridorigin(x::Grid) = Vec(map(first, gridaxes(x)))

checkshapefunction(::Grid{<: Any, <: Any, Nothing}) = throw(ArgumentError("`Grid` must include the information of shape function, see help `?Grid` for more details."))
checkshapefunction(::Grid{<: Any, <: Any, <: ShapeFunction}) = nothing

function Grid(::Type{Node}, shapefunction, coordinates::Coordinate) where {Node}
    state = SpArray(StructVector{Node}(undef, 0), SpPattern(size(coordinates)))
    axes = coordinateaxes(coordinates)
    Grid(shapefunction, Coordinate(Array.(axes)), map(step, axes), state)
end

Grid(::Type{Node}, shapefunction, axes::Tuple{Vararg{AbstractVector}}) where {Node} = Grid(Node, shapefunction, Coordinate(axes))
Grid(::Type{Node}, axes::Tuple{Vararg{AbstractVector}}) where {Node} = Grid(Node, nothing, axes)
Grid(shapefunction, axes::Tuple{Vararg{AbstractVector}}) = Grid(Nothing, shapefunction, axes)
Grid(axes::Tuple{Vararg{AbstractVector}}) = Grid(Nothing, nothing, axes)

Grid(Node::Type, shapefunction, axes::AbstractVector...) = Grid(Node, shapefunction, axes)
Grid(Node::Type, axes::AbstractVector...) = Grid(Node, nothing, axes)
Grid(shapefunction, axes::AbstractVector...) = Grid(Nothing, shapefunction, axes)
Grid(axes::AbstractVector...) = Grid(Nothing, nothing, axes)

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
6-element Grid{1, Float64, Nothing, Nothing, Poingr.SpArray{Nothing, 1, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}, Tuple{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}}}:
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
    inds = CartesianIndices(@. UnitRange(unsafe_trunc(Int,  ceil(ξ - h))+1,
                                         unsafe_trunc(Int, floor(ξ + h))+1))
    CartesianIndices(grid) ∩ inds
end
@inline function neighboring_nodes(grid::Grid, x::Vec)
    checkshapefunction(grid)
    neighboring_nodes(grid, x, support_length(grid.shapefunction))
end

function update!(gridindices::Vector{Index{dim}}, grid::Grid{dim}, x::Vec{dim}, spat::BitArray{dim}) where {dim}
    inds = neighboring_nodes(grid, x)
    cnt = 0
    @inbounds @simd for I in inds
        cnt += ifelse(spat[I], 1, 0)
    end
    resize!(gridindices, cnt)
    cnt = 0
    @inbounds for I in inds
        if spat[I]
            gridindices[cnt+=1] = Index(grid, I)
        end
    end
    gridindices
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
6×6 Grid{2, Float64, Nothing, Nothing, Poingr.SpArray{Nothing, 2, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}, Tuple{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}}}:
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
6×6 Grid{2, Float64, Nothing, Nothing, Poingr.SpArray{Nothing, 2, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}, Tuple{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}}}:
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
11×11 Grid{2, Float64, Nothing, Nothing, Poingr.SpArray{Nothing, 2, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}, Tuple{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}}}:
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
    @inbounds Threads.@threads for p in eachindex(xₚ)
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
    spat_threads = [fill(false, size(grid)) for _ in 1:Threads.nthreads()]
    @inbounds Threads.@threads for x in xₚ
        spat = spat_threads[Threads.threadid()]
        inds = neighboring_nodes(grid, x, 1)
        spat[inds] .= true
    end
    broadcast(|, spat_threads...)
end

# this seems to be faster when using `reordering_pointstate!`
function sparsity_pattern(grid::Grid, xₚ::AbstractVector, ptsinblk::AbstractArray{Vector{Int}})
    spat = falses(size(grid))
    for color in coloringblocks(size(grid))
        Threads.@threads for blockindex in color
            @inbounds for p in ptsinblk[blockindex]
                inds = neighboring_nodes(grid, xₚ[p], 1)
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

nfill(v, ::Val{N}) where {N} = ntuple(d -> v, Val(N))
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
