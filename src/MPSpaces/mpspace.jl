struct MPSpace{dim, FT <: ShapeFunction{dim}, GT <: AbstractGrid{dim}, VT <: ShapeValue{dim}}
    F::FT
    grid::GT
    dofmap::DofMap{dim}
    dofindices::Vector{Vector{Int}}
    dofindices_dim::Vector{Vector{Int}}
    gridindices::Vector{Vector{CartesianIndex{dim}}}
    Ns::PointState{VT}
end

function MPSpace(::Type{T}, F::ShapeFunction{dim}, grid::AbstractGrid{dim}, npoints::Int) where {dim, T}
    dofmap = DofMap(size(grid))
    dofindices = [Int[] for _ in 1:npoints]
    dofindices_dim = [Int[] for _ in 1:npoints]
    gridindices = [CartesianIndex{dim}[] for _ in 1:npoints]
    Ns = PointState([construct(T, F) for _ in 1:npoints])
    MPSpace(F, grid, dofmap, dofindices, dofindices_dim, gridindices, Ns)
end

function MPSpace(F::ShapeFunction, grid::AbstractGrid, npoints::Int)
    MPSpace(Float64, F, grid, npoints)
end

function reinit_dofmap!(space::MPSpace{dim}, coordinates::AbstractArray{<: Vec{dim}}; exclude = nothing, point_radius::Real) where {dim}
    dofindices = space.dofindices
    dofindices_dim = space.dofindices_dim
    gridindices = space.gridindices
    @assert length(coordinates) == length(dofindices) == length(dofindices_dim) == length(gridindices)

    grid = space.grid
    dofmap = space.dofmap

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
        DofHelpers.map!(dofmap, dofindices[i], allinds)
        DofHelpers.map!(dofmap, dofindices_dim[i], allinds; dof = dim)
        DofHelpers.filter!(dofmap, gridindices[i], allinds)
    end

    space
end

function reinit_shapevalue!(space::MPSpace, coordinates::AbstractArray{<: Vec})
    for (j, (x, N)) in enumerate(zip(coordinates, space.Ns))
        inds = gridindices(space, j)
        reinit!(N, space.grid, inds, x)
    end
    space
end

function reinit!(space::MPSpace, coordinates::AbstractArray{<: Vec}; exclude = nothing)
    point_radius = Interpolations.support_length(space.F)
    reinit_dofmap!(space, coordinates; point_radius, exclude)
    reinit_shapevalue!(space, coordinates)
    space
end

"""
    dofindices(::MPSpace, p::Int; [dof])

Return dof indices at point index `p`.
Use [`reinit!(::MPSpace, ::AbstractArray{<: Vec})`](@ref) in advance.
"""
function dofindices(space::MPSpace{dim}, p::Int; dof::Int = 1) where {dim}
    @_propagate_inbounds_meta
    dof == 1   && return space.dofindices[p]
    dof == dim && return space.dofindices_dim[p]
    DofHelpers.map(space.dofmap, space.gridindices[p])
end

"""
    gridindices(::MPSpace, p::Int)

Return grid indices at point index `p`.
Use [`reinit!(::MPSpace, ::AbstractArray{<: Vec})`](@ref) in advance.
"""
gridindices(space::MPSpace, p::Int) = (@_propagate_inbounds_meta; space.gridindices[p])

"""
    ndofs(::MPSpace; [dof])

Return total number of dofs.
"""
ndofs(space::MPSpace; dof::Int = 1) = ndofs(space.dofmap; dof)

npoints(space::MPSpace) = length(space.dofindices)


function from_point_to_grid!(S::SparseArray, space::MPSpace, f)
    @assert indices(S) === indices(space.dofmap)
    ∑ = f(space.Ns)
    nzval = nonzeros(zeros!(S))
    @inbounds for p in 1:npoints(space)
        v = view(nzval, dofindices(space, p))
        v .+= ∑[p]
    end
    S
end

function from_point_to_grid(space::MPSpace, f)
    ∑ = f(space.Ns)
    ElType = eltype(∑[1])
    S = SparseArray(typeofzero(ElType), space.dofmap) # typeofzero is to handle ScalarVector or VectorTensor values
    from_point_to_grid!(S, space, f)
end

typeofzero(x) = typeof(zero(x))
