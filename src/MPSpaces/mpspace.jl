struct MPSpace{dim, T, FT <: ShapeFunction{dim}, GT <: AbstractGrid{dim}, VT <: ShapeValue{dim}}
    F::FT
    grid::GT
    dofmap::DofMap{dim}
    dofindices::Vector{Vector{Int}}
    dofindices_dim::Vector{Vector{Int}}
    gridindices::Vector{Vector{CartesianIndex{dim}}}
    Nᵢ::PointState{VT}
    uᵢ::SparseArray{dim, T}
    uₚ::PointState{T}
end

function MPSpace(::Type{T}, F::ShapeFunction{dim}, grid::AbstractGrid{dim}, npoints::Int) where {dim, T <: Union{Real, Vec}}
    dofmap = DofMap(size(grid))
    dofindices = [Int[] for _ in 1:npoints]
    dofindices_dim = [Int[] for _ in 1:npoints]
    gridindices = [CartesianIndex{dim}[] for _ in 1:npoints]
    Nᵢ = PointState([construct(eltype(T), F) for _ in 1:npoints])
    uᵢ = SparseArray(T, dofmap)
    uₚ = zeros!(PointState(T, npoints))
    MPSpace(F, grid, dofmap, dofindices, dofindices_dim, gridindices, Nᵢ, uᵢ, uₚ)
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
    for (j, (x, N)) in enumerate(zip(coordinates, space.Nᵢ))
        inds = gridindices(space, j)
        reinit!(N, space.grid, inds, x)
    end
    space
end

function reinit!(space::MPSpace, coordinates::AbstractArray{<: Vec}; exclude = nothing)
    point_radius = ShapeFunctions.support_length(space.F)
    reinit_dofmap!(space, coordinates; point_radius, exclude)
    reinit_shapevalue!(space, coordinates)
    zeros!(space.uᵢ)
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


#################
# point_to_grid #
#################

function _point_to_grid!(S::SparseArray, space::MPSpace, ∑ₚwu::SumToGrid)
    @assert indices(S) === indices(space.dofmap)
    nzval = nonzeros(zeros!(S))
    @inbounds for p in 1:npoints(space)
        u = view(nzval, dofindices(space, p))
        u .+= ∑ₚwu[p]
    end
    S
end

function _point_to_grid(space::MPSpace, ∑ₚwu::SumToGrid)
    ElType = eltype(∑ₚwu[1])
    S = SparseArray(typeofzero(ElType), space.dofmap) # typeofzero is to handle ScalarVector or VectorTensor values
    _point_to_grid!(S, space, ∑ₚwu)
end

function point_to_grid!(S::SparseArray, space::MPSpace, f)
    ∑ₚwu = f(space.Nᵢ)
    _point_to_grid!(S, space, ∑ₚwu)
end

function point_to_grid(space::MPSpace, f)
    ∑ₚwu = f(space.Nᵢ)
    _point_to_grid(space, ∑ₚwu)
end

#################
# grid_to_point #
#################

struct PointToGridMap{T} <: AbstractCollection{2, T}
    data::Vector{T}
    dofindices::Vector{Vector{Int}}
end

p2gmap(space::MPSpace, S::SparseArray) = PointToGridMap(nonzeros(S), space.dofindices)

Base.length(v::PointToGridMap) = length(v.dofindices) # == npoints
Base.getindex(v::PointToGridMap, i::Int) = (@_propagate_inbounds_meta; Collection{1}(view(v.data, v.dofindices[i])))

function value_gradient(Nᵢ, uᵢ)
    ∑ᵢ(ValueGradient(uᵢ * Nᵢ, _otimes_(uᵢ, ∇(Nᵢ))))
end

function _grid_to_point!(dest::PointState, space::MPSpace, ∑ᵢwu::SumToPoint)
    @assert length(dest) == npoints(space)
    @inbounds for p in 1:npoints(space)
        dest[p] = ∑ᵢwu[p]
    end
    dest
end

function _grid_to_point(space::MPSpace, ∑ᵢwu::SumToPoint)
    ElType = typeof(∑ᵢwu[1])
    dest = PointState(ElType, npoints(space)) # typeofzero is to handle ScalarVector or VectorTensor values
    _grid_to_point!(dest, space, ∑ᵢwu)
end

# with f

function grid_to_point!(dest::PointState, space::MPSpace, src::SparseArray, f)
    @assert indices(src) == indices(space.dofmap)
    Nᵢ = space.Nᵢ
    uᵢ = p2gmap(space, src)
    ∑ᵢwu = f(Nᵢ, uᵢ)
    _point_to_grid!(dest, space, ∑ᵢwu)
end

function grid_to_point(space::MPSpace, src::SparseArray, f)
    @assert indices(src) === indices(space.dofmap)
    Nᵢ = space.Nᵢ
    uᵢ = p2gmap(space, src)
    ∑ᵢwu = f(Nᵢ, uᵢ)
    _grid_to_point(space, ∑ᵢwu)
end

# without f

function grid_to_point!(dest::PointState, space::MPSpace, src::SparseArray)
    @assert indices(src) === indices(space.dofmap)
    Nᵢ = space.Nᵢ
    uᵢ = p2gmap(space, src)
    uₚ = value_gradient(Nᵢ, uᵢ)
    _grid_to_point!(dest, space, uₚ)
end

function grid_to_point(space::MPSpace, src::SparseArray)
    @assert indices(src) === indices(space.dofmap)
    Nᵢ = space.Nᵢ
    uᵢ = p2gmap(space, src)
    uₚ = value_gradient(Nᵢ, uᵢ)
    _grid_to_point(space, uₚ)
end

################
# grid_to_grid #
################

function grid_to_grid!(dest::SparseArray, space::MPSpace, src::SparseArray, f)
    @assert indices(src) === indices(space.dofmap)
    Nᵢ = space.Nᵢ
    uᵢ = p2gmap(space, src)
    uₚ = value_gradient(Nᵢ, uᵢ)
    point_to_grid!(dest, space, N -> f(N, uₚ))
end

function grid_to_grid(space::MPSpace, src::SparseArray, f)
    @assert indices(src) === indices(space.dofmap)
    Nᵢ = space.Nᵢ
    uᵢ = p2gmap(space, src)
    uₚ = value_gradient(Nᵢ, uᵢ)
    point_to_grid(space, N -> f(N, uₚ))
end

function grid_to_grid(space::MPSpace, f)
    grid_to_grid(space, space.uᵢ, f)
end

###########################
# function_reconstruction #
###########################

function function_reconstruction!(wu_i::SparseArray, w_i::SparseArray, space::MPSpace, u::PointState, w = identity) # identity means w(N) = N
    point_to_grid!(wu_i, space, N -> ∑ₚ(u*w(N)))
    point_to_grid!(w_i, space, N -> ∑ₚ(w(N)))
    nonzeros(wu_i) ./= nonzeros(w_i)
    wu_i
end

function function_reconstruction!(wu_i::SparseArray, space::MPSpace, u::PointState, w = identity) # identity means w(N) = N
    point_to_grid!(wu_i, space, N -> ∑ₚ(u*w(N)))
    w_i = point_to_grid(space, N -> ∑ₚ(w(N)))
    nonzeros(wu_i) ./= nonzeros(w_i)
    wu_i
end

function function_reconstruction(space::MPSpace, u::PointState, w = identity) # identity means w(N) = N
    wu_i = point_to_grid(space, N -> ∑ₚ(u*w(N)))
    w_i = point_to_grid(space, N -> ∑ₚ(w(N)))
    nonzeros(wu_i) ./= nonzeros(w_i)
    wu_i
end


typeofzero(x) = typeof(zero(x))
