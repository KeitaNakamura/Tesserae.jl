struct PointState{T} <: AbstractCollection{2}
    data::Vector{T}
end

pointstate(data::Vector) = PointState(data)
pointstate(::Type{T}, length) where {T} = pointstate([zero(T) for i in 1:length])
pointstate(c::AbstractCollection{2}) = (p = pointstate(eltype(c), length(c)); p ← c)

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

set!(p::PointState, c::Union{AbstractCollection{2}, AbstractVector}) = (p.data .= c; p)
function set!(p::PointState, c::Union{AbstractCollection{2}, AbstractVector}, activepoints::BitVector)
    @assert length(p) == length(c) == length(activepoints)
    for i in 1:length(p)
        if activepoints[i]
            @inbounds p.data[i] = c[i]
        end
    end
    p
end
@generated function set!(ps::Tuple{Vararg{PointState, N}}, c::Union{AbstractCollection{2}, AbstractVector}) where {N}
    exps = [:(ps[$i][p] = x[$i]) for i in 1:N]
    quote
        (@nall $N i -> length(ps[1]) == length(ps[i])) || error("length must match")
        first(c) isa Tuple{Vararg{Any, N}} || throw(ArgumentError("types must match"))
        @inbounds for p in 1:length(c)
            x = c[p]
            $(exps...)
        end
    end
end
const ← = set!

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
