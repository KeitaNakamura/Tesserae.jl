###############
# Grid states #
###############

function generate_gridstate(GridState::Type, dims::Dims)
    SpArray(StructVector{GridState}(undef, 0), SpPattern(dims), true, Ref(NaN))
end
generate_gridstate(GridState::Type, dims::Int...) = generate_gridstate(GridState, dims)
generate_gridstate(GridState::Type, grid::AbstractArray) = generate_gridstate(GridState, size(grid))

################
# Point states #
################

function generate_pointstate(isindomain::Function, PointState::Type, grid::Grid{dim, T}; n::Int = 2) where {dim, T}
    axes = gridaxes(grid)
    dims = size(grid)
    h = gridsteps(grid) ./ n # length per particle
    allpoints = Grid(@. LinRange(first(axes)+h/2, last(axes)-h/2, n*(dims-1)))

    # find points `isindomain`
    mask = broadcast(x -> isindomain(x...), allpoints)
    npts = count(mask)

    pointstate = StructVector{PointState}(undef, npts)
    fillzero!(pointstate)

    if :x in propertynames(pointstate)
        cnt = 0
        @inbounds for (i, x) in enumerate(allpoints)
            if mask[i]
                pointstate.x[cnt+=1] = x
            end
        end
    end
    if :V in propertynames(pointstate)
        V = prod(h)
        if dim == 2 && grid.coordinate_system isa Axisymmetric
            @. pointstate.V = getindex(pointstate.x, 1) * V
        else
            @. pointstate.V = V
        end
    end
    if :r in propertynames(pointstate)
        pointstate.r .= Vec(h) / 2
    end
    if :index in propertynames(pointstate)
        pointstate.index .= 1:npts
    end

    reorder_pointstate!(pointstate, pointsperblock(grid, pointstate.x))
    pointstate
end

function generate_pointstate(isindomain::Function, grid::Grid{dim, T}; kwargs...) where {dim, T}
    PointState = @NamedTuple begin
        x::Vec{dim, T}
        V::T
        r::Vec{dim, T}
        index::Int
    end
    generate_pointstate(isindomain, PointState, grid; kwargs...)
end
