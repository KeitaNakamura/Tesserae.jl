const GridBound{dim} = NamedTuple{(:index, :component, :n), Tuple{CartesianIndex{dim}, Int, Vec{dim, Int}}}

"""
    GridBoundSet(indices::CartesianIndices{dim}, direction::Int)

Create `GridBoundSet` with `indices`.
`i = abs(direction)` represents `i`th component such as `1 ≤ i ≤ dim`.
`sign(direction)` represents the direction of unit normal vector.
The element of `GridBoundSet` has fields `:index`, `:component` and `:n`.

# Examples
```jldoctest
julia> GridBoundSet(CartesianIndices((1:2,1:2)), 2)
GridBoundSet{2} with 4 elements:
  (index = CartesianIndex(1, 1), component = 2, n = [0, 1])
  (index = CartesianIndex(2, 1), component = 2, n = [0, 1])
  (index = CartesianIndex(1, 2), component = 2, n = [0, 1])
  (index = CartesianIndex(2, 2), component = 2, n = [0, 1])
```
"""
struct GridBoundSet{dim} <: AbstractSet{GridBound{dim}}
    set::Set{GridBound{dim}}
end

function GridBoundSet(inds::CartesianIndices{dim}, direction::Int) where {dim}
    n = Vec{dim, Int}(d -> d == abs(direction) ? sign(direction) : 0)
    GridBoundSet(Set([GridBound{dim}((i, abs(direction), n)) for i in inds]))
end

Base.length(bound::GridBoundSet) = length(bound.set)
Base.emptymutable(::GridBoundSet{T}, ::Type{U} = T) where {T, U} = GridBoundSet(Set{U}())

Base.iterate(bound::GridBoundSet, state...) = iterate(bound.set, state...)
Base.push!(bound::GridBoundSet, v) = push!(bound.set, v)


abstract type AbstractGrid{dim, T} <: AbstractArray{Vec{dim, T}, dim} end

Base.size(grid::AbstractGrid) = size(getaxisarray(grid))
gridsteps(grid::AbstractGrid) = map(step, gridaxes(grid))
gridsteps(grid::AbstractGrid, i::Int) = gridsteps(grid)[i]
gridaxes(grid::AbstractGrid) = parent(getaxisarray(grid))
gridaxes(grid::AbstractGrid, i::Int) = (@_propagate_inbounds_meta; gridaxes(grid)[i])

@inline function Base.getindex(grid::AbstractGrid{dim}, I::Vararg{Int, dim}) where {dim}
    @boundscheck checkbounds(grid, I...)
    @inbounds Vec(getaxisarray(grid)[I...])
end

@inline function Base.getindex(grid::AbstractGrid, I::Vararg{Union{AbstractUnitRange, Colon}})
    @boundscheck checkbounds(grid, I...)
    @inbounds newgrid = typeof(grid)(getaxisarray(grid)[I...])
    unionboundsets!(newgrid, grid)
    newgrid
end

@inline function Base.getindex(grid::AbstractGrid, I::CartesianIndices)
    @boundscheck checkbounds(grid, I)
    @inbounds getindex(grid, I.indices...)
end

function unionboundsets!(dest::AbstractGrid{dim}, src::AbstractGrid{dim}) where {dim}
    for (name, srcset) in getboundsets(src)
        destset = get!(getboundsets(dest), name, Set{GridBound{dim}}())
        union!(destset, [bound for bound in srcset if checkbounds(Bool, dest, bound.index)])
    end
    dest
end

gridorigin(grid::AbstractGrid) = map(first, gridaxes(grid))
getboundsets(grid::AbstractGrid) = grid.boundsets
getboundset(grid::AbstractGrid, name::String) = getboundsets(grid)[name]
getboundset(grid::AbstractGrid) = union(values(getboundsets(grid))...)
setboundset!(grid::AbstractGrid, name::String, set::GridBoundSet) = getboundsets(grid)[name] = set

function generate_default_boundsets!(grid::AbstractGrid{1})
    start = firstindex(grid, 1)
    stop = lastindex(grid, 1)
    setboundset!(grid, "-x", GridBoundSet(CartesianIndices((start:start,)), 1))
    setboundset!(grid, "+x", GridBoundSet(CartesianIndices((stop:stop,)), -1))
    grid
end

function generate_default_boundsets!(grid::AbstractGrid{2})
    start = firstindex.((grid,), (1, 2))
    stop = lastindex.((grid,), (1, 2))
    range = UnitRange.(start, stop)
    setboundset!(grid, "-x", GridBoundSet(CartesianIndices((start[1], range[2])), 1))
    setboundset!(grid, "-y", GridBoundSet(CartesianIndices((range[1], start[2])), 2))
    setboundset!(grid, "+x", GridBoundSet(CartesianIndices((stop[1], range[2])), -1))
    setboundset!(grid, "+y", GridBoundSet(CartesianIndices((range[1], stop[2])), -2))
    grid
end

function generate_default_boundsets!(grid::AbstractGrid{3})
    start = firstindex.((grid,), (1, 2, 3))
    stop = lastindex.((grid,), (1, 2, 3))
    range = UnitRange.(start, stop)
    setboundset!(grid, "-x", GridBoundSet(CartesianIndices((start[1], range[2], range[3])), 1))
    setboundset!(grid, "-y", GridBoundSet(CartesianIndices((range[1], start[2], range[3])), 2))
    setboundset!(grid, "-z", GridBoundSet(CartesianIndices((range[1], range[2], start[3])), 3))
    setboundset!(grid, "-x", GridBoundSet(CartesianIndices((stop[1], range[2], range[3])), -1))
    setboundset!(grid, "-y", GridBoundSet(CartesianIndices((range[1], stop[2], range[3])), -2))
    setboundset!(grid, "-z", GridBoundSet(CartesianIndices((range[1], range[2], stop[3])), -3))
    grid
end

"""
    CartesianGrid(origin, dx::Real, ncells)
    CartesianGrid(dx::Real, ncells)

Create cartesian-grid with `ncells` and linear space `dx`.
If `origin` is not given, it is set to zero.
Default boundary sets storing `GridBound` are also generated, which are `"-x"`, `"+x"`, `"-y"`, `"+y"`, `"-z"` and `"+z"` for each dimension.

# Examples
```jldoctest
julia> grid = CartesianGrid(1.0, (5, 5))
6×6 CartesianGrid{2,Float64}:
 [0.0, 0.0]  [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]  [0.0, 5.0]
 [1.0, 0.0]  [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]  [1.0, 5.0]
 [2.0, 0.0]  [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]  [2.0, 5.0]
 [3.0, 0.0]  [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]  [3.0, 5.0]
 [4.0, 0.0]  [4.0, 1.0]  [4.0, 2.0]  [4.0, 3.0]  [4.0, 4.0]  [4.0, 5.0]
 [5.0, 0.0]  [5.0, 1.0]  [5.0, 2.0]  [5.0, 3.0]  [5.0, 4.0]  [5.0, 5.0]

julia> all(bd -> bd.n == [1, 0], getboundset(grid, "-x"))
true

julia> grid == CartesianGrid((0, 0), 1.0, 5)
true
```
"""
struct CartesianGrid{dim, T} <: AbstractGrid{dim, T}
    axisarray::CartesianAxisArray{dim, T}
    boundsets::Dict{String, Set{GridBound{dim}}}
end

# from CartesianAxisArray
function CartesianGrid{dim, T}(axisarray::CartesianAxisArray{dim, T}) where {dim, T}
    CartesianGrid{dim, T}(axisarray, Dict())
end
CartesianGrid{dim}(axisarray::CartesianAxisArray{dim, T}) where {dim, T} = CartesianGrid{dim, T}(axisarray)
CartesianGrid(axisarray::CartesianAxisArray{dim, T}) where {dim, T} = CartesianGrid{dim, T}(axisarray)

for params in ((:dim, :T), (:dim,), ())
    @eval begin
        # with origin
        function CartesianGrid{$(params...)}(origin::Union{Real, Tuple}, dx::Real, ncells::Union{Int, Tuple}) where {$(params...)}
            axisarray = CartesianAxisArray{$(params...)}(origin, dx, ncells .+ 1)
            grid = CartesianGrid{$(params...)}(axisarray)
            generate_default_boundsets!(grid)
            grid
        end
        # without origin
        function CartesianGrid{$(params...)}(dx::Real, ncells::Union{Int, Tuple}) where {$(params...)}
            axisarray = CartesianAxisArray{$(params...)}(0, dx, ncells .+ 1)
            grid = CartesianGrid{$(params...)}(axisarray)
            generate_default_boundsets!(grid)
            grid
        end
    end
end

getaxisarray(grid::CartesianGrid) = grid.axisarray


"""
    Grid(axes::AbstractVector...)

Construct `Grid` by `axes`.

# Examples
```jldoctest
julia> grid = Grid(range(0, 3, step = 1.0), range(1, 4, step = 1.0))
4×4 Grid{2,Float64,StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}}:
 [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]
 [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]
 [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]
 [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]
```
"""
struct Grid{dim, T, V} <: AbstractGrid{dim, T}
    axisarray::AxisArray{dim, T, V}
    boundsets::Dict{String, Set{GridBound{dim}}}
end

function Grid{dim, T, V}(axisarray::AxisArray{dim, T, V}) where {dim, T, V}
    Grid{dim, T, V}(axisarray, Dict())
end

function Grid(axs::NTuple{dim, V}) where {dim, T, V <: AbstractVector{T}}
    grid = Grid{dim, T, V}(AxisArray(axs))
    generate_default_boundsets!(grid)
    grid
end

Grid(axs::Vararg{AbstractVector}) = Grid(axs)

getaxisarray(grid::Grid) = grid.axisarray
