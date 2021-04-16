struct MPSpace{dim, FT <: ShapeFunction{dim}, GT <: AbstractGrid{dim}, VT <: ShapeValue{dim}}
    F::FT
    grid::GT
    dofmap::DofMap{dim}
    dofindices::PointToDofIndices
    gridindices::PointToGridIndices{dim}
    activeindices::Vector{CartesianIndex{dim}}
    freedofs::Vector{Int} # flat dofs
    bounddofs::Vector{Int}
    nearsurface::BitVector
    Nᵢ::PointState{VT}
    # for threads
    ptranges::Vector{UnitRange{Int}}
    dofmap_threads::Vector{DofMap{dim}}
    dofindices_threads::Vector{PointToDofIndices}
end

function chunk_ranges(total::Int, nchunks::Int)
    splits = [round(Int, s) for s in range(0, stop=total, length=nchunks+1)]
    [splits[i]+1:splits[i+1] for i in 1:nchunks]
end

function MPSpace(::Type{T}, F::ShapeFunction{dim}, grid::AbstractGrid{dim}, npoints::Int) where {dim, T <: Real}
    dofmap = DofMap(size(grid))

    construct_dofindices(n) = [Int[] for _ in 1:n]

    dofindices = construct_dofindices(npoints)
    gridindices = [CartesianIndex{dim}[] for _ in 1:npoints]

    activeindices = CartesianIndex{dim}[]
    freedofs = Int[]
    bounddofs = Int[]
    nearsurface = falses(npoints)
    Nᵢ = pointstate([construct(T, F) for _ in 1:npoints])

    ptranges = chunk_ranges(npoints, Threads.nthreads())
    dofmap_threads = [DofMap(size(grid)) for i in 1:Threads.nthreads()]
    dofindices_threads = [construct_dofindices(length(ptranges[i])) for i in 1:Threads.nthreads()]

    MPSpace(F, grid, dofmap, dofindices, gridindices, activeindices, freedofs, bounddofs, nearsurface, Nᵢ,
            ptranges, dofmap_threads, dofindices_threads)
end

MPSpace(F::ShapeFunction, grid::AbstractGrid, npoints::Int) = MPSpace(Float64, F, grid, npoints)

value_gradient_type(::Type{T}, ::Val{dim}) where {T <: Real, dim} = ScalVec{dim, T}
value_gradient_type(::Type{Vec{dim, T}}, ::Val{dim}) where {T, dim} = VecTensor{dim, T, dim^2}

function reinit_threads!(space, coordinates, exclude, point_radius)
    grid = space.grid
    gridindices = space.gridindices

    id = Threads.threadid()
    ptrange = space.ptranges[id]
    coords = @view coordinates[ptrange]
    dofmap = space.dofmap_threads[id]
    dofindices = space.dofindices_threads[id]
    @assert length(coords) == length(dofindices)

    # Reinitialize dofmap

    ## reset
    dofmap .= false

    ## activate grid indices and store them.
    for x in coords
        inds = neighboring_nodes(grid, x, point_radius)
        @inbounds dofmap[inds] .= true
    end

    ## exclude grid nodes if a function is given
    if exclude !== nothing
        # if exclude(xi) is true, then make it false
        @inbounds for i in eachindex(grid, dofmap)
            if dofmap[i] == true
                xi = grid[i]
                exclude(xi) && (dofmap[i] = false)
            end
        end
        # surrounding nodes are activated
        for x in coords
            inds = neighboring_nodes(grid, x, 1)
            @inbounds dofmap[inds] .= true
        end
    end

    ## renumering dofs
    count!(dofmap)

    # Reinitialize shape values and dof indices by updated DofMap
    @inbounds for (i, p) in enumerate(ptrange)
        allinds = neighboring_nodes(grid, coords[i], point_radius)
        DofHelpers.map!(dofmap, dofindices[i], allinds)
        allactive = DofHelpers.filter!(dofmap, gridindices[p], allinds)
        space.nearsurface[p] = !allactive
    end
end

function reinit!(space::MPSpace{dim}, coordinates; exclude = nothing) where {dim}
    point_radius = ShapeFunctions.support_length(space.F)

    @assert length(space.gridindices) == length(coordinates)

    Threads.@threads for _ in 1:Threads.nthreads()
        reinit_threads!(space, coordinates, exclude, point_radius)
    end

    # update global dofmap and dofindices
    space.dofmap .= false
    broadcast!(|, space.dofmap, space.dofmap_threads...)
    count!(space.dofmap)
    @inbounds for p in 1:length(coordinates)
        DofHelpers.map!(space.dofmap, space.dofindices[p], space.gridindices[p])
    end

    ## active grid indices
    DofHelpers.filter!(space.dofmap, space.activeindices, CartesianIndices(space.dofmap))

    ## freedofs (used in dirichlet boundary conditions)
    # TODO: modify for scalar field: need to create freedofs for scalar field?
    empty!(space.freedofs)
    @inbounds for i in CartesianIndices(space.dofmap)
        I = space.dofmap(i; dof = dim)
        I === nothing && continue
        if onbound(space.dofmap, i)
            for d in 1:dim
                if !onbound(size(space.dofmap, d), i[d])
                    push!(space.freedofs, I[d])
                end
            end
        else
            append!(space.freedofs, I)
        end
    end

    ## bounddofs (!!NOT!! flat)
    empty!(space.bounddofs)
    @inbounds for i in CartesianIndices(space.dofmap)
        I = space.dofmap(i)
        I === nothing && continue
        if onbound(space.dofmap, i)
            push!(space.bounddofs, I)
        end
    end

    Threads.@threads for i in 1:Threads.nthreads()
        rng = space.ptranges[Threads.threadid()]
        @inbounds for p in rng
            inds = gridindices(space, p)
            reinit!(space.Nᵢ[p], space.grid, inds, coordinates[p])
        end
    end

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
    GridStateThreads(gridstate(T, space.dofmap, space.dofindices),
                     space.ptranges,
                     [gridstate(T, dofmap, dofindices) for (dofmap, dofindices) in zip(space.dofmap_threads, space.dofindices_threads)])
end

function gridstate_matrix(space::MPSpace, T)
    gridstate_matrix(T, space.dofindices, space.freedofs)
end

function pointstate(space::MPSpace, T)
    pointstate(T, npoints(space))
end

function construct(name::Symbol, space::MPSpace)
    name == :shape_value        && return space.Nᵢ
    name == :shape_vector_value && return lazy(vec, space.Nᵢ)
    if name == :bound_normal_vector
        A = BoundNormalArray(Float64, gridsize(space)...)
        return GridState(SparseArray(view(A, space.activeindices), space.dofmap), space.dofindices)
    end
    if name == :grid_coordinates
        return GridStateCollection(view(space.grid, space.activeindices), space.dofindices)
    end
    if eltype(space.Nᵢ) <: WLSValue
        if name == :weight_value
            return lazy(ShapeFunctions.weight_value, space.Nᵢ)
        end
        if name == :moment_matrix_inverse
            return lazy(ShapeFunctions.moment_matrix_inverse, space.Nᵢ)
        end
    end
    throw(ArgumentError("$name in $(space.F) is not supported"))
end

function dirichlet!(vᵢ::GridState{dim, Vec{dim, T}}, space::MPSpace{dim}) where {dim, T}
    V = reinterpret(T, nonzeros(vᵢ))
    fixeddofs = setdiff(1:ndofs(space, dof = dim), space.freedofs)
    V[fixeddofs] .= zero(T)
    vᵢ
end
