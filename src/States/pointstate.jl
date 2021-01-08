struct PointState{T} <: AbstractCollection{2, T}
    data::Vector{T}
end

pointstate(data::Vector) = PointState(data)
pointstate(::Type{T}, length) where {T} = pointstate(zeros(T, length))
pointstate(c::UnionCollection{2}) = (p = pointstate(eltype(c), length(c)); p ← c)

Base.length(p::PointState) = length(p.data)
Base.Array(p::PointState) = p.data

@inline function Base.getindex(p::PointState, i::Int)
    @boundscheck checkbounds(p, i)
    @inbounds p.data[i]
end

@inline function Base.setindex!(p::PointState, v, i::Int)
    @boundscheck checkbounds(p, i)
    @inbounds p.data[i] = v
end

Base.fill!(p::PointState, v) = fill!(p.data, v)

Base.similar(p::PointState, ::Type{T}) where {T} = pointstate(T, length(p))
Base.similar(p::PointState{T}) where {T} = similar(p, T)

# left arrow

set!(p::PointState, c::UnionCollection{2}) = (p.data .= c; p)
set!(p::PointState, v::AbstractVector) = (p.data .= v; p)
const ← = set!

# colon computation

isrank2(x::Type{<: UnionCollection}) = x <: UnionCollection{2} || throw(ArgumentError("support only rank=2 collections"))
isrank2(x) = false

addref(x::UnionCollection{2}) = x
addref(x) = Ref(x)

(::Colon)(op, x::UnionCollection{2}) = lazy(op, x)

@generated function (::Colon)(op, xs::Tuple)
    any(isrank2, xs.parameters) ?
        :(LazyCollection{2}(broadcasted(op, map(addref, xs)...))) :
        :(throw(ArgumentError("no rank=2 collections")))
end

# generate point states

function generate_pointstates(indomain, grid::AbstractGrid{dim, T}, coordinate_system = :plane_strain_if_2D; n::Int = 2) where {dim, T}
    h = gridsteps(grid) ./ n # length per particle
    allpoints = Grid(StepRangeLen.(first.(gridaxes(grid)) .+ h./2, h, n .* (size(grid) .- 1) .- 1))
    npoints = count(x -> indomain(x...), allpoints)
    xₚ = pointstate(Vec{dim, T}, npoints)
    Vₚ = pointstate(T, npoints)
    hₚ = pointstate(Vec{dim, T}, npoints)
    i = 0
    for x in allpoints
        if indomain(x...)
            xₚ[i+=1] = x
        end
    end
    V = prod(h)
    for i in 1:npoints
        if dim == 2 && coordinate_system == :axisymmetric
            r = xₚ[i][1]
            Vₚ[i] = r * V
        else
            Vₚ[i] = V
        end
        hₚ[i] = Vec(h)
    end
    (; xₚ, Vₚ, hₚ)
end
