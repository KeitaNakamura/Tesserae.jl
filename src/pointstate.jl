default_pointstate_type(::Nothing, ::Val{dim}, ::Val{T}) where {dim, T} =
    @NamedTuple{x::Vec{dim, T}, V::T, r::Vec{dim, T}, index::Int}

struct DefaultPointState{dim, T}
    m::T
    V::T
    x::Vec{dim, T}
    v::Vec{dim, T}
    r::Vec{dim, T}
    b::Vec{dim, T}
    σ::SymmetricSecondOrderTensor{3, T, 6}
    ϵ::SymmetricSecondOrderTensor{3, T, 6}
    ∇v::SecondOrderTensor{3, T, 9}
    index::Int
end

default_pointstate_type(::BSpline, ::Val{dim}, ::Val{T}) where {dim, T} = DefaultPointState{dim, T}
default_pointstate_type(::GIMP, ::Val{dim}, ::Val{T}) where {dim, T} = DefaultPointState{dim, T}

struct DefaultPointStateWLS{dim, T, L, dim_L}
    m::T
    V::T
    x::Vec{dim, T}
    v::Vec{dim, T}
    r::Vec{dim, T}
    b::Vec{dim, T}
    σ::SymmetricSecondOrderTensor{3, T, 6}
    ϵ::SymmetricSecondOrderTensor{3, T, 6}
    ∇v::SecondOrderTensor{3, T, 9}
    C::Mat{dim, L, T, dim_L}
    index::Int
end

default_pointstate_type(::LinearWLS, ::Val{dim}, ::Val{T}) where {dim, T} = DefaultPointStateWLS{dim, T, dim+1, dim*(dim+1)}
default_pointstate_type(::BilinearWLS, ::Val{2}, ::Val{T}) where {T} = DefaultPointStateWLS{2, T, 4, 8}

function generate_pointstate(indomain, Point::Type, grid::Grid{dim, T}; n::Int = 2) where {dim, T}
    h = gridsteps(grid) ./ n # length per particle
    allpoints = Grid(@. LinRange(
        first($gridaxes(grid)) + h/2,
        last($gridaxes(grid))  - h/2,
        n * ($size(grid) - 1)
    ))

    npoints = count(x -> indomain(x...), allpoints)
    pointstate = StructVector{Point}(undef, npoints)
    for name in propertynames(pointstate)
        v = getproperty(pointstate, name)
        fillzero!(v)
    end

    if :x in propertynames(pointstate)
        cnt = 0
        for x in allpoints
            if indomain(x...)
                @inbounds pointstate.x[cnt+=1] = x
            end
        end
    end
    if :V in propertynames(pointstate)
        V = prod(h)
        if dim == 2 && grid.coordinate_system == :axisymmetric
            @. pointstate.V = getindex(pointstate.x, 1) * V
        else
            @. pointstate.V = V
        end
    end
    if :r in propertynames(pointstate)
        pointstate.r .= Vec(h) / 2
    end
    if :index in propertynames(pointstate)
        pointstate.index .= 1:npoints
    end

    pointstate
end

function generate_pointstate(indomain, grid::Grid{dim, T}; kwargs...) where {dim, T}
    Point = default_pointstate_type(grid.shapefunction, Val(dim), Val(T))
    generate_pointstate(indomain, Point, grid; kwargs...)
end

function remove_pointstate_outside_domain!(pointstate, grid::Grid)
    inds = findall(pointstate.x) do x
        @inbounds begin
            !(grid[begin][1] ≤ x[1] ≤ grid[end][1] &&
              grid[begin][2] ≤ x[2] ≤ grid[end][2])
        end
    end
    StructArrays.foreachfield(v -> deleteat!(v, inds), pointstate)
    pointstate
end
