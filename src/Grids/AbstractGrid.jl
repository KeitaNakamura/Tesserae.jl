abstract type AbstractGrid{dim, T} <: AbstractArray{Vec{dim, T}, dim} end

Base.size(grid::AbstractGrid) = size(getaxisarray(grid))
gridsteps(grid::AbstractGrid) = map(step, gridaxes(grid))
gridsteps(grid::AbstractGrid, i::Int) = gridsteps(grid)[i]
gridaxes(grid::AbstractGrid) = coordinateaxes(getaxisarray(grid))
gridaxes(grid::AbstractGrid, i::Int) = (@_propagate_inbounds_meta; gridaxes(grid)[i])

@inline function Base.getindex(grid::AbstractGrid{dim}, I::Vararg{Int, dim}) where {dim}
    @boundscheck checkbounds(grid, I...)
    @inbounds Vec(getaxisarray(grid)[I...])
end

@inline function Base.getindex(grid::AbstractGrid, I::Vararg{Union{AbstractUnitRange, Colon}})
    @boundscheck checkbounds(grid, I...)
    @inbounds newgrid = typeof(grid)(getaxisarray(grid)[I...])
    newgrid
end

@inline function Base.getindex(grid::AbstractGrid, I::CartesianIndices)
    @boundscheck checkbounds(grid, I)
    @inbounds getindex(grid, I.indices...)
end

gridorigin(grid::AbstractGrid) = map(first, gridaxes(grid))
