"""
    Grid(axes::AbstractVector...)
    Grid{dim}(axis::AbstractVector)

Construct `Grid` by `axes`.

# Examples
```jldoctest
julia> Grid(range(0, 3, step = 1.0), range(1, 4, step = 1.0))
4×4 Grid{2, Float64, Tuple{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}}, Nothing, Poingr.MaskedArray{Nothing, 2, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}}:
 [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]
 [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]
 [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]
 [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]
```
"""
struct Grid{dim, T, Axes, Node, State <: MaskedArray{Node, dim}} <: AbstractArray{Vec{dim, T}, dim}
    coordinates::Coordinate{dim, NTuple{dim, T}, Axes}
    state::State
end

Base.size(x::Grid) = size(x.coordinates)
gridsteps(x::Grid) = map(step, gridaxes(x))
gridsteps(x::Grid, i::Int) = gridsteps(x)[i]
gridaxes(x::Grid) = coordinateaxes(x.coordinates)
gridaxes(x::Grid, i::Int) = (@_propagate_inbounds_meta; gridaxes(x)[i])
gridorigin(x::Grid) = map(first, gridaxes(x))

function Grid(::Type{Node}, axes::Tuple{Vararg{AbstractVector}}) where {Node}
    coordinates = Coordinate(axes)
    state = MaskedArray(StructVector{Node}(undef, 0), Mask(size(coordinates)))
    Grid(coordinates, state)
end
Grid(Node::Type, axes::AbstractVector...) = Grid(Node, axes)
Grid(axes::Tuple{Vararg{AbstractVector}}) = Grid(Nothing, axes)
Grid(axes::AbstractVector...) = Grid(axes)

function Grid{dim}(::Type{Node}, axis::AbstractVector) where {dim, Node}
    coordinates = Coordinate{dim}(axis)
    state = MaskedArray(StructVector{Node}(undef, 0), Mask(size(coordinates)))
    Grid(coordinates, state)
end
Grid{dim}(axis::AbstractVector) where {dim} = Grid{dim}(Nothing, axis)

@inline function Base.getindex(grid::Grid{dim}, i::Vararg{Int, dim}) where {dim}
    @boundscheck checkbounds(grid, i...)
    @inbounds Vec(grid.coordinates[i...])
end

function neighboring_nodes(ax::AbstractVector, x::Real, h::Real)
    xmin = first(ax)
    xmax = last(ax)
    xmin ≤ x ≤ xmax || return 1:0
    ξ = (x - xmin) / step(ax)
    _neighboring_nodes(ξ, h, length(ax))
end

function _neighboring_nodes(ξ::Real, h::Real, len::Int)
    ξ_l = ξ - h
    ξ_r = ξ + h
    l = ceil(ξ_l)
    r = floor(ξ_r)
    l === ξ_l && (l += 1)
    r === ξ_r && (r -= 1)
    start, stop = clamp.((Int(l+1), Int(r+1)), 1, len) # cut violated indices
    start:stop
end

"""
    Poingr.neighboring_nodes(grid, x::Vec, h::Real)

Return `CartesianIndices` storing neighboring node indices around `x`.
`h` is a range for searching and its unit is `gridsteps` `dx`.
In 1D, for example, the searching range becomes `x ± h*dx`.

# Examples
```jldoctest
julia> grid = Grid(0.0:1.0:5.0)
6-element Grid{1, Float64, Tuple{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}}, Nothing, Poingr.MaskedArray{Nothing, 1, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}}:
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
@generated function neighboring_nodes(grid::Grid{dim}, x::Vec{dim}, h::Real) where {dim}
    quote
        @_inline_meta
        @inbounds CartesianIndices(
            @ntuple $dim d -> begin
                ax = gridaxes(grid, d)
                neighboring_nodes(ax, x[d], h)
            end
        )
    end
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
6×6 Grid{2, Float64, Tuple{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}}, Nothing, Poingr.MaskedArray{Nothing, 2, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}}:
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
@generated function neighboring_cells(grid::Grid{dim}, cellindex::CartesianIndex{dim}, h::Int) where {dim}
    quote
        @boundscheck checkbounds(grid, cellindex + oneunit(cellindex))
        @inbounds CartesianIndices(
            @ntuple $dim d -> begin
                i = cellindex[d]
                len = size(grid, d) - 1
                start, stop = clamp.((i-h, i+h), 1, len) # cut violated indices
                start:stop
            end
        )
    end
end

@inline function neighboring_cells(grid::Grid{dim}, x::Vec{dim}, h::Int) where {dim}
    neighboring_cells(grid, whichcell(grid, x), h)
end

"""
    Poingr.whichcell(grid, x::Vec)

Return cell index where `x` locates.

# Examples
```jldoctest
julia> grid = Grid(0.0:1.0:5.0, 0.0:1.0:5.0)
6×6 Grid{2, Float64, Tuple{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}}, Nothing, Poingr.MaskedArray{Nothing, 2, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}}:
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
@generated function whichcell(grid::Grid{dim}, x::Vec{dim}) where {dim}
    quote
        ncells = size(grid) .- 1
        dx = gridsteps(grid)
        xmin = gridorigin(grid)
        xmax = xmin .+ dx .* ncells
        @inbounds begin
            (@nall $dim d -> xmin[d] ≤ x[d] ≤ xmax[d]) || return nothing
            CartesianIndex(@ntuple $dim d -> floor(Int, (x[d] - xmin[d]) / dx[d]) + 1)
        end
    end
end

function pointsinblock!(ptsinblk::AbstractArray{Vector{Int}, dim}, grid::Grid{dim}, xₚ::AbstractVector) where {dim}
    empty!.(ptsinblk)
    for p in eachindex(xₚ)
        I = whichcell(grid, xₚ[p])
        I === nothing && continue
        blockindex = @. ($Tuple(I)-1) >> BLOCK_UNIT + 1
        push!(ptsinblk[blockindex...], p)
    end
    ptsinblk
end

function pointsinblock(grid::Grid, xₚ::AbstractVector)
    blocksize = @. (($size(grid) - 1) - 1) >> BLOCK_UNIT + 1
    ptsinblk = [Int[] for i in CartesianIndices(blocksize)]
    pointsinblock!(ptsinblk, grid, xₚ)
end


struct BlockIndices{N} <: AbstractArray{CartesianIndex{N}, N}
    inds::Coordinate{N, NTuple{N, Int}, NTuple{N, StepRange{Int, Int}}}
end
Base.size(x::BlockIndices) = size(x.inds)
Base.getindex(x::BlockIndices{N}, i::Vararg{Int, N}) where {N} = (@_propagate_inbounds_meta; CartesianIndex(x.inds[i...]))

nfill(v, ::Val{N}) where {N} = ntuple(d -> v, Val(N))
function coloringblocks(dims::NTuple{dim, Int}) where {dim}
    ncells = dims .- 1
    starts = SArray{NTuple{dim, 2}}(Iterators.ProductIterator(nfill((1,2), Val(dim)))...)
    nblocks = @. (ncells - 1) >> BLOCK_UNIT + 1
    vec(map(st -> BlockIndices(Coordinate(StepRange.(st, 2, nblocks))), starts))
end

function generate_pointstate(indomain, Point::Type, grid::Grid{dim, T}, coordinate_system = :plane_strain_if_2D; n::Int = 2) where {dim, T}
    h = gridsteps(grid) ./ n # length per particle
    allpoints = Grid(LinRange.(first.(gridaxes(grid)) .+ h./2, last.(gridaxes(grid)) .- h./2, n .* (size(grid) .- 1)))

    npoints = count(x -> indomain(x...), allpoints)
    pointstate = reinit!(StructVector{Point}(undef, npoints))

    cnt = 0
    for x in allpoints
        if indomain(x...)
            pointstate.x[cnt+=1] = x
        end
    end

    V = prod(h)
    for i in 1:npoints
        if dim == 2 && coordinate_system == :axisymmetric
            r = pointstate.x[i][1]
            pointstate.V0[i] = r * V
        else
            pointstate.V0[i] = V
        end
        pointstate.h[i] = Vec(h)
    end

    pointstate
end
function generate_pointstate(indomain, grid::Grid{dim, T}, coordinate_system = :plane_strain_if_2D; n::Int = 2) where {dim, T}
    generate_pointstate(indomain, @NamedTuple{x::Vec{dim, T}, V0::T, h::Vec{dim,T}}, grid, coordinate_system; n)
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

struct GridIndex{dim}
    i::Int
    I::CartesianIndex{dim}
end
@inline GridIndex(grid, i::Int) = (@_propagate_inbounds_meta; GridIndex(i, CartesianIndices(grid)[i]))
@inline GridIndex(grid, I::CartesianIndex) = (@_propagate_inbounds_meta; GridIndex(LinearIndices(grid)[I], I))
@inline Base.checkindex(::Type{Bool}, inds::AbstractUnitRange, i::GridIndex) = checkindex(Bool, inds, i.i)
@inline _to_indices(::IndexLinear, A, inds, I::Tuple{GridIndex, Vararg{Any}}) = to_indices(A, inds, (I[1].i, Base.tail(I)...))
@inline _to_indices(::IndexCartesian, A, inds, I::Tuple{GridIndex, Vararg{Any}}) = to_indices(A, inds, (Tuple(I[1].I)..., Base.tail(I)...))
@inline Base.to_indices(A, inds, I::Tuple{GridIndex, Vararg{Any}}) = _to_indices(IndexStyle(A), A, inds, I)
