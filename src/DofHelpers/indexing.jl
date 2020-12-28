"""
    PointToGridIndex(npoints::Int, grid::AbstractGrid)

Construct object handling point-to-grid indices.
See also [`numbering!(::PointToGridIndex, ::AbstractArray{<: Vec})`](@ref).
"""
struct PointToGridIndex{spacedim, G <: AbstractGrid{spacedim}}
    grid::G
    dofmap::DofMap{spacedim}
    dofindices::Vector{Vector{Int}}
    dofindices_dim::Vector{Vector{Int}}
    gridindices::Vector{Vector{CartesianIndex{spacedim}}}
end

function PointToGridIndex(npoints::Int, grid::AbstractGrid{spacedim}) where {spacedim}
    dofmap = DofMap(size(grid))
    dofindices = [Int[] for _ in 1:npoints]
    dofindices_dim = [Int[] for _ in 1:npoints]
    gridindices = [CartesianIndex{spacedim}[] for _ in 1:npoints]
    PointToGridIndex(grid, dofmap, dofindices, dofindices_dim, gridindices)
end

"""
    numbering!(::PointToGridIndex, coordinates::AbstractArray{<: Vec}; point_radius::Real, exclude = nothing)

Numbering point-to-dof and point-to-grid indices by `coordinates` of points.
`point_radius` is `h` in [`neighboring_nodes(grid, x::Vec, h::Real)`](@ref).

# Examples
```jldoctest
julia> points = [Vec(0.5, 0.5), Vec(5.0, 3.0)];

julia> pg = PointToGridIndex(length(points), Grid(0:4, 0:3));

julia> numbering!(pg, points; point_radius = 1);

julia> dofindices(pg, 1)
4-element Array{Int64,1}:
 1
 2
 3
 4

julia> dofindices(pg, 2)
Int64[]

julia> gridindices(pg, 1)
4-element Array{CartesianIndex{2},1}:
 CartesianIndex(1, 1)
 CartesianIndex(2, 1)
 CartesianIndex(1, 2)
 CartesianIndex(2, 2)

julia> gridindices(pg, 2)
CartesianIndex{2}[]
```
"""
function numbering!(pg::PointToGridIndex{dim}, coordinates::AbstractArray{<: Vec{dim}}; exclude = nothing, point_radius::Real) where {dim}
    dofindices = pg.dofindices
    dofindices_dim = pg.dofindices_dim
    gridindices = pg.gridindices
    @assert length(coordinates) == length(dofindices) == length(dofindices_dim) == length(gridindices)

    grid = pg.grid
    dofmap = pg.dofmap

    # Reinitialize dofmap

    ## reset
    dofmap .= false

    ## activate grid indices and store them.
    for x in coordinates
        inds = neighboring_nodes(grid, x, point_radius)
        @inbounds dofmap[inds] .= true
    end

    ## exclude grid nodes if a function is given
    if exclude !== nothing
        # if exclude(xi) is true, then make it false
        @inbounds for i in eachindex(grid)
            xi = grid[i]
            exclude(xi) && (dofmap[i] = false)
        end
        # surrounding nodes are activated
        for x in coordinates
            inds = neighboring_nodes(grid, x, 1)
            @inbounds dofmap[inds] .= true
        end
    end

    ## renumering dofs
    count!(dofmap)

    # Reinitialize interpolations by updated mask
    @inbounds for (i, x) in enumerate(coordinates)
        allinds = neighboring_nodes(grid, x, point_radius)
        map!(dofmap, dofindices[i], allinds)
        map!(dofmap, dofindices_dim[i], allinds; dim)
        filter!(dofmap, gridindices[i], allinds)
    end

    pg
end

"""
    dofindices(::PointToGridIndex, p::Int)
    dofindices(::PointToGridIndex, p::Int, dim::Int)

Return dof indices at point index `p`.
Use [`numbering!(::PointToGridIndex, ::AbstractArray{<: Vec})`](@ref) in advance.
"""
dofindices(pg::PointToGridIndex, p::Int) = (@_propagate_inbounds_meta; pg.dofindices[p])

function dofindices(pg::PointToGridIndex{spacedim}, p::Int, dim::Int) where {spacedim}
    @_propagate_inbounds_meta
    spacedim == 1   && return pg.dofindices[p]
    spacedim == dim && return pg.dofindices_dim[p]
    map(pg.dofmap, pg.gridindices[p])
end

"""
    gridindices(::PointToGridIndex, p::Int)

Return grid indices at point index `p`.
Use [`numbering!(::PointToGridIndex, ::AbstractArray{<: Vec})`](@ref) in advance.
"""
gridindices(pg::PointToGridIndex, p::Int) = (@_propagate_inbounds_meta; pg.gridindices[p])

"""
    ndofs(::PointToGridIndex)

Return total number of dofs.
"""
ndofs(pg::PointToGridIndex) = ndofs(pg.dofmap)
