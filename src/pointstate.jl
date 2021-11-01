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
    dϵ_v::T
    index::Int
end

default_pointstate_type(::BSpline, ::Val{dim}, ::Val{T}) where {dim, T} = DefaultPointState{dim, T}
default_pointstate_type(::GIMP, ::Val{dim}, ::Val{T}) where {dim, T} = DefaultPointState{dim, T}

struct DefaultPointStateWLS{dim, T, M}
    m::T
    V::T
    x::Vec{dim, T}
    v::Vec{dim, T}
    r::Vec{dim, T}
    b::Vec{dim, T}
    σ::SymmetricSecondOrderTensor{3, T, 6}
    ϵ::SymmetricSecondOrderTensor{3, T, 6}
    ∇v::SecondOrderTensor{3, T, 9}
    C::Mat{dim, M, T, 6}
    dϵ_v::T
    index::Int
end

default_pointstate_type(::LinearWLS, ::Val{dim}, ::Val{T}) where {dim, T} = DefaultPointStateWLS{dim, T, dim+1}

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
        ElType = eltype(v)
        if isbitstype(ElType)
            v .= initval(ElType)
        end
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
