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
