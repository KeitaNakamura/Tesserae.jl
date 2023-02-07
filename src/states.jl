using PoissonDiskSampling

###############
# Grid states #
###############

@generated function generate_gridstate(::Type{GridState}, dims::Dims) where {GridState}
    exps = [:(SpArray{$T}(sppat, stamp)) for T in fieldtypes(GridState)]
    quote
        sppat = SpPattern(dims)
        stamp = Ref(NaN)
        StructArray{GridState}(tuple($(exps...)))
    end
end
generate_gridstate(GridState::Type, dims::Int...) = generate_gridstate(GridState, dims)
generate_gridstate(GridState::Type, grid::AbstractArray) = generate_gridstate(GridState, size(grid))

################
# Point states #
################

function generate_points_regularly(grid::Grid, n::Int)
    axes = gridaxes(grid)
    dims = size(grid)
    r = gridsteps(grid) ./ 2n
    Grid((@. LinRange(first(axes)+r, last(axes)-r, n*(dims-1)))...)
end

function generate_points_randomly(grid::Grid, n::Int)
    d = gridsteps(grid) ./ n
    minmaxes = map((min,max)->(min,max), Tuple(first(grid)), Tuple(last(grid)))
    points = PoissonDiskSampling.generate(minmaxes...; r = only(unique(d)))
    map(eltype(grid), points)
end

function generate_pointstate(isindomain::Function, ::Type{PointState}, grid::Grid{dim}; n::Int = 2, random::Bool = false) where {PointState, dim}
    if random
        allpoints = generate_points_randomly(grid, n)
    else
        allpoints = generate_points_regularly(grid, n)
    end
    V = prod(last(grid) - first(grid)) / length(allpoints)

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
        if dim == 2 && gridsystem(grid) isa Axisymmetric
            @. pointstate.V = getindex(pointstate.x, 1) * V
        else
            @. pointstate.V = V
        end
    end
    if :r in propertynames(pointstate)
        if random
            h = ones(Vec{dim}) * V^(1/dim)
        else
            h = Vec(gridsteps(grid) ./ n)
        end
        pointstate.r .= h / 2
    end
    if :index in propertynames(pointstate)
        pointstate.index .= 1:npts
    end

    reorder_pointstate!(pointstate, pointsperblock(grid, pointstate.x))
    pointstate
end

function generate_pointstate(isindomain::Function, grid::Grid{dim, T}; kwargs...) where {dim, T}
    PointState = minimum_pointstate_type(Val(dim), T)
    generate_pointstate(isindomain, PointState, grid; kwargs...)
end

function generate_pointstate(::Type{PointState}, pointstate_old::StructVector) where {PointState}
    pointstate = StructVector{PointState}(undef, length(pointstate_old))
    fillzero!(pointstate)

    if :x in propertynames(pointstate)
        pointstate.x .= pointstate_old.x
    end
    if :V in propertynames(pointstate)
        pointstate.V .= pointstate_old.V
    end
    if :r in propertynames(pointstate)
        pointstate.r .= pointstate_old.r
    end
    if :index in propertynames(pointstate)
        pointstate.index .= pointstate_old.index
    end

    pointstate
end

function generate_pointstate(pointstate_old::StructVector)
    T = eltype(pointstate_old.x)
    PointState = minimum_pointstate_type(Val(length(T)), eltype(T))
    generate_pointstate(PointState, pointstate_old)
end

function minimum_pointstate_type(::Val{dim}, ::Type{T}) where {dim, T}
    @NamedTuple begin
        x::Vec{dim, T}
        V::T
        r::Vec{dim, T}
        index::Int
    end
end
