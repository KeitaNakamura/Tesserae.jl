struct MPSpace{dim, FT <: ShapeFunction{dim}, GT <: AbstractGrid{dim}, VT <: ShapeValue{dim}}
    F::FT
    grid::GT
    dofmap::DofMap{dim}
    dofindices::Vector{Vector{Int}}
    gridindices::Vector{Vector{CartesianIndex{dim}}}
    activeindices::Vector{CartesianIndex{dim}}
    fixeddofs::Vector{Int} # flat dofs
    bounddofs::Vector{Int}
    isincontact::BitVector
    Nᵢ::PointState{VT}
end

function MPSpace(::Type{T}, F::ShapeFunction{dim}, grid::AbstractGrid{dim}, npoints::Int) where {dim, T <: Real}
    dofmap = DofMap(size(grid))
    dofindices = [Int[] for _ in 1:npoints]
    gridindices = [CartesianIndex{dim}[] for _ in 1:npoints]
    activeindices = CartesianIndex{dim}[]
    fixeddofs = Int[]
    bounddofs = Int[]
    isincontact = falses(npoints)
    Nᵢ = pointstate([construct(T, F) for _ in 1:npoints])
    MPSpace(F, grid, dofmap, dofindices, gridindices, activeindices, fixeddofs, bounddofs, isincontact, Nᵢ)
end

MPSpace(F::ShapeFunction, grid::AbstractGrid, npoints::Int) = MPSpace(Float64, F, grid, npoints)

value_gradient_type(::Type{T}, ::Val{dim}) where {T <: Real, dim} = ScalVec{dim, T}
value_gradient_type(::Type{Vec{dim, T}}, ::Val{dim}) where {T, dim} = VecTensor{dim, T, dim^2}

function reinit_dofmap!(space::MPSpace{dim}, coordinates; exclude = nothing, point_radius::Real) where {dim}
    dofindices = space.dofindices
    gridindices = space.gridindices
    @assert length(coordinates) == length(dofindices) == length(gridindices)

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

    # Initialize dof indices by updated DofMap
    @inbounds for (i, x) in enumerate(coordinates)
        allinds = neighboring_nodes(grid, x, point_radius)
        DofHelpers.map!(dofmap, dofindices[i], allinds)
        allactive = DofHelpers.filter!(dofmap, gridindices[i], allinds)
        space.isincontact[i] = !allactive
    end

    ## active grid indices
    DofHelpers.filter!(dofmap, space.activeindices, CartesianIndices(dofmap))

    ## fixeddofs (used in dirichlet boundary conditions)
    count = 0
    empty!(space.fixeddofs)
    @inbounds for i in CartesianIndices(dofmap)
        I = dofmap(i; dof = dim)
        I === nothing && continue
        if onbound(dofmap, i)
            for d in 1:dim
                if onbound(size(dofmap, d), i[d])
                    push!(space.fixeddofs, I[d])
                    count += 1
                end
            end
        end
    end
    resize!(space.fixeddofs, count)

    ## bounddofs (!!NOT!! flat)
    count = 0
    empty!(space.bounddofs)
    @inbounds for i in CartesianIndices(dofmap)
        I = dofmap(i)
        I === nothing && continue
        if onbound(dofmap, i)
            push!(space.bounddofs, I)
            count += 1
        end
    end
    resize!(space.bounddofs, count)

    space
end

function reinit_shapevalue!(space::MPSpace, coordinates)
    @inbounds for (j, (x, N)) in enumerate(zip(coordinates, space.Nᵢ))
        inds = gridindices(space, j)
        reinit!(N, space.grid, inds, x)
    end
    space
end

function reinit!(space::MPSpace, coordinates; exclude = nothing)
    point_radius = ShapeFunctions.support_length(space.F)
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
    DofHelpers.map(space.dofmap, space.gridindices[p]; dof)
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
gridsize(space::MPSpace) = size(space.grid)

function gridstate(space::MPSpace, T)
    gridstate(T, space.dofmap, space.dofindices)
end

function gridstate_matrix(space::MPSpace, T)
    gridstate_matrix(T, space.dofindices)
end

function pointstate(space::MPSpace, T)
    pointstate(T, npoints(space))
end

function construct(name::Symbol, space::MPSpace)
    name == :shape_function        && return space.Nᵢ
    name == :shape_function_vector && return lazy(vec, space.Nᵢ)
    if name == :bound_normal_vector
        boundnormal = BoundNormal(Float64, gridsize(space)...)
        return gridstate(view(boundnormal, space.activeindices), space.dofmap, space.dofindices)
    end
    throw(ArgumentError("not supported function space"))
end

function dirichlet!(vᵢ::GridState{dim, Vec{dim, T}}, space::MPSpace{dim}) where {dim, T}
    V = reinterpret(T, nonzeros(vᵢ))
    V[space.fixeddofs] .= zero(T)
    vᵢ
end
