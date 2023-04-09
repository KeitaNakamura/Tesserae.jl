struct AxisArray{dim, T, V <:AbstractVector{T}} <: AbstractArray{NTuple{dim, T}, dim}
    axes::NTuple{dim, V}
end
get_axes(A::AxisArray) = A.axes
Base.size(A::AxisArray) = map(length, A.axes)
@generated function Base.getindex(A::AxisArray{dim}, i::Vararg{Integer, dim}) where {dim}
    quote
        @_inline_meta
        @_propagate_inbounds_meta
        @ntuple $dim d -> A.axes[d][i[d]]
    end
end
@inline function Base.getindex(A::AxisArray{dim}, ranges::Vararg{AbstractUnitRange{Int}, dim}) where {dim}
    @_propagate_inbounds_meta
    AxisArray(map(getindex, A.axes, ranges))
end

struct LinAxis{T} <: AbstractVector{T}
    dx::T
    xmin::T
    len::Int
end
function LinAxis{T}(dx::Real, (xmin,xmax)::Tuple{Real, Real}) where {T}
    len = length(range(xmin, xmax; step=dx))
    LinAxis{T}(dx, xmin, len)
end
function LinAxis(dx::Real, (xmin, xmax)::Tuple{Real, Real})
    T = promote_type(typeof(dx), typeof(xmin), typeof(xmax))
    LinAxis{T}(dx, (xmin, xmax))
end
Base.size(x::LinAxis) = (x.len,)
@inline function Base.getindex(x::LinAxis, i::Integer)
    @boundscheck checkbounds(x, i)
    @inbounds muladd(x.dx, i-one(i), x.xmin)
end

"""
    Lattice(dx, (xmin, xmax), (ymin, ymax)...)
    Lattice(T, dx, (xmin, xmax), (ymin, ymax)...)

Construct `Lattice` with the spacing `dx`.

# Examples
```jldoctest
julia> Lattice(1.0, (0,3), (1,4))
4×4 Lattice{2, Float64, Vector{Float64}}:
 [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]
 [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]
 [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]
 [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]
```
"""
struct Lattice{dim, T, V <: AbstractVector{T}} <: AbstractArray{Vec{dim, T}, dim}
    axisarray::AxisArray{dim, T, V}
    dx::T
    dx_inv::T
end

get_axisarray(x::Lattice) = x.axisarray
Base.size(x::Lattice) = size(get_axisarray(x))
# helpers
spacing(x::Lattice) = x.dx
spacing_inv(x::Lattice) = x.dx_inv
get_axes(x::Lattice) = get_axes(x.axisarray)
get_axes(x::Lattice, i::Integer) = (@_propagate_inbounds_meta; get_axes(x)[i])

function Lattice(::Type{T}, dx::Real, minmax::Vararg{Tuple{Real, Real}, dim}) where {T, dim}
    @assert all(map(issorted, minmax))
    axisarray = AxisArray(LinAxis{T}.(dx, minmax))
    Lattice(axisarray, T(dx), T(inv(dx)))
end
Lattice(dx::Real, minmax::Tuple{Real, Real}...) = Lattice(Float64, dx, minmax...)

@inline function Base.getindex(lattice::Lattice{dim}, i::Vararg{Integer, dim}) where {dim}
    @boundscheck checkbounds(lattice, i...)
    @inbounds Vec(get_axisarray(lattice)[i...])
end
@inline function Base.getindex(lattice::Lattice{dim}, ranges::Vararg{AbstractUnitRange{Int}, dim}) where {dim}
    @boundscheck checkbounds(lattice, ranges...)
    @inbounds Lattice(get_axisarray(lattice)[ranges...], spacing(lattice), spacing_inv(lattice))
end

@inline function isinside(x::Vec, lattice::Lattice{dim, T}) where {dim, T}
    axes = get_axes(lattice)
    @inbounds _isinside(SVec{dim, T}(map(first, axes)), SVec{dim, T}(map(last, axes)), SVec{dim, T}(x))
end
@inline function _isinside(start::SVec{dim, T}, stop::SVec{dim, T}, x::SVec{dim, T}) where {dim, T}
    all(start ≤ x) && all(x ≤ stop)
end

"""
    neighbornodes(lattice, x::Vec, h) -> (indices, isfullyinside)

Return `CartesianIndices` storing neighboring node `indices` around `x`.
`h` denotes the range for searching area. In 1D, for example, the range `a`
becomes ` x-h*dx < a < x+h*dx` where `dx` is `spacing(lattice)`.
`isfullyinside` is `true` if the neighboring nodes are completely inside of
the `lattice`.

# Examples
```jldoctest
julia> lattice = Lattice(1, (0,5))
6-element Lattice{1, Float64, Vector{Float64}}:
 [0.0]
 [1.0]
 [2.0]
 [3.0]
 [4.0]
 [5.0]

julia> neighbornodes(lattice, Vec(1.5), 1)
(CartesianIndices((2:3,)), true)

julia> neighbornodes(lattice, Vec(1.5), 3)
(CartesianIndices((1:5,)), false)
```
"""
@inline function neighbornodes(lattice::Lattice{dim, T}, x::Vec, h::Real) where {dim, T}
    isinside(x, lattice) || return (CartesianIndices(nfill(1:0, Val(dim))), false)
    _neighborindices(SVec{dim,Int}(size(lattice)), spacing_inv(lattice), SVec{dim,T}(first(lattice)), SVec{dim,T}(x), convert(T, h))
end
@inline function _neighborindices(dims::SVec{dim, Int}, dx⁻¹::T, xmin::SVec{dim, T}, x::SVec{dim, T}, h::T) where {dim, T}
    ξ = (x - xmin) * dx⁻¹
    start = convert(SVec{dim, Int}, floor(ξ - h)) + 2
    stop  = convert(SVec{dim, Int}, floor(ξ + h)) + 1
    imin = Tuple(max(start, 1))
    imax = Tuple(min(stop, dims))
    isfullyinside = all(1 ≤ start) && all(stop ≤ dims)
    CartesianIndices(UnitRange.(imin, imax)), isfullyinside
end

"""
    Marble.whichcell(lattice, x::Vec)

Return cell index where `x` locates.

# Examples
```jldoctest
julia> lattice = Lattice(1, (0,5), (0,5))
6×6 Lattice{2, Float64, Vector{Float64}}:
 [0.0, 0.0]  [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]  [0.0, 5.0]
 [1.0, 0.0]  [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]  [1.0, 5.0]
 [2.0, 0.0]  [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]  [2.0, 5.0]
 [3.0, 0.0]  [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]  [3.0, 5.0]
 [4.0, 0.0]  [4.0, 1.0]  [4.0, 2.0]  [4.0, 3.0]  [4.0, 4.0]  [4.0, 5.0]
 [5.0, 0.0]  [5.0, 1.0]  [5.0, 2.0]  [5.0, 3.0]  [5.0, 4.0]  [5.0, 5.0]

julia> Marble.whichcell(lattice, Vec(1.5, 1.5))
CartesianIndex(2, 2)
```
"""
@inline function whichcell(lattice::Lattice{dim, T}, x::Vec) where {dim, T}
    isinside(x, lattice) || return nothing
    _whichcell(spacing_inv(lattice), SVec{dim, T}(first(lattice)), SVec{dim, T}(x))
end
@inline function _whichcell(dx⁻¹::T, xmin::SVec{dim, T}, x::SVec{dim, T}) where {dim, T}
    ξ = (x - xmin) * dx⁻¹
    CartesianIndex(Tuple(convert(SVec{dim, Int}, floor(ξ)) + 1))
end
