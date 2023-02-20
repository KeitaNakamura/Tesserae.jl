struct AxisArray{dim, T, V <:AbstractVector{T}} <: AbstractArray{NTuple{dim, T}, dim}
    axes::NTuple{dim, V}
end
get_axes(A::AxisArray) = A.axes
Base.size(A::AxisArray) = map(length, A.axes)
@generated function Base.getindex(A::AxisArray{dim}, i::Vararg{Int, dim}) where {dim}
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

"""
    Lattice(dx, (xmin, xmax), (ymin, ymax)...)
    Lattice(T, dx, (xmin, xmax), (ymin, ymax)...)

Construct `Lattice` with the spacing `dx`.

# Examples
```jldoctest
julia> Lattice(1.0, (0,3), (1,4))
4×4 Lattice{2, Float64}:
 [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]
 [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]
 [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]
 [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]
```
"""
struct Lattice{dim, T} <: AbstractArray{Vec{dim, T}, dim}
    axisarray::AxisArray{dim, T, Vector{T}}
    dx::T
    dx_inv::T
end

get_axisarray(x::Lattice) = x.axisarray
Base.size(x::Lattice) = size(get_axisarray(x))
# helpers
spacing(x::Lattice) = x.dx
spacing_inv(x::Lattice) = x.dx_inv
get_axes(x::Lattice) = get_axes(x.axisarray)
get_axes(x::Lattice, i::Int) = (@_propagate_inbounds_meta; get_axes(x)[i])

function Lattice(::Type{T}, dx::Real, minmax::Vararg{Tuple{Real, Real}, dim}) where {T, dim}
    @assert all(map(issorted, minmax))
    axes = map(x->range(x...; step=dx), minmax)
    axisarray = AxisArray(map(Vector{T}, axes))
    Lattice(axisarray, T(dx), T(inv(dx)))
end
Lattice(dx::Real, minmax::Tuple{Real, Real}...) = Lattice(Float64, dx, minmax...)

@inline function Base.getindex(lattice::Lattice{dim}, i::Vararg{Int, dim}) where {dim}
    @boundscheck checkbounds(lattice, i...)
    @inbounds Vec(get_axisarray(lattice)[i...])
end
@inline function Base.getindex(lattice::Lattice{dim}, ranges::Vararg{AbstractUnitRange{Int}, dim}) where {dim}
    @boundscheck checkbounds(lattice, ranges...)
    @inbounds Lattice(get_axisarray(lattice)[ranges...], spacing(lattice), spacing_inv(lattice))
end

@generated function isinside(x::Vec{dim}, lattice::Lattice{dim}) where {dim}
    quote
        @_inline_meta
        @inbounds begin
            @nexprs $dim i -> start_i = get_axes(lattice, i)[begin]
            @nexprs $dim i -> stop_i  = get_axes(lattice, i)[end]
            @nall $dim i -> start_i ≤ x[i] ≤ stop_i
        end
    end
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
6-element Lattice{1, Float64}:
 [0.0]
 [1.0]
 [2.0]
 [3.0]
 [4.0]
 [5.0]

julia> neighbornodes(lattice, Vec(1.5), 1)
(CartesianIndices((2:3,)), false)

julia> neighbornodes(lattice, Vec(1.5), 3)
(CartesianIndices((1:5,)), true)
```
"""
@inline function neighbornodes(lattice::Lattice{dim, T}, x::Vec{dim, T}, h::Real) where {dim, T}
    isinside(x, lattice) || return (CartesianIndices(nfill(1:0, Val(dim))), false)
    _neighborindices(SVec(size(lattice)), spacing_inv(lattice), SVec(first(lattice)), SVec(x), convert(T, h))
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
6×6 Lattice{2, Float64}:
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
@inline function whichcell(lattice::Lattice, x::Vec)
    isinside(x, lattice) || return nothing
    dx⁻¹ = spacing_inv(lattice)
    xmin = first(lattice)
    ξ = Tuple((x - xmin) * dx⁻¹)
    CartesianIndex(@. unsafe_trunc(Int, floor(ξ)) + 1)
end

"""
    Marble.whichblock(lattice, x::Vec)

Return block index where `x` locates.
The unit block size is `2^$BLOCK_UNIT` cells.

# Examples
```jldoctest
julia> lattice = Lattice(1, (0,10), (0,10))
11×11 Lattice{2, Float64}:
 [0.0, 0.0]   [0.0, 1.0]   [0.0, 2.0]   …  [0.0, 9.0]   [0.0, 10.0]
 [1.0, 0.0]   [1.0, 1.0]   [1.0, 2.0]      [1.0, 9.0]   [1.0, 10.0]
 [2.0, 0.0]   [2.0, 1.0]   [2.0, 2.0]      [2.0, 9.0]   [2.0, 10.0]
 [3.0, 0.0]   [3.0, 1.0]   [3.0, 2.0]      [3.0, 9.0]   [3.0, 10.0]
 [4.0, 0.0]   [4.0, 1.0]   [4.0, 2.0]      [4.0, 9.0]   [4.0, 10.0]
 [5.0, 0.0]   [5.0, 1.0]   [5.0, 2.0]   …  [5.0, 9.0]   [5.0, 10.0]
 [6.0, 0.0]   [6.0, 1.0]   [6.0, 2.0]      [6.0, 9.0]   [6.0, 10.0]
 [7.0, 0.0]   [7.0, 1.0]   [7.0, 2.0]      [7.0, 9.0]   [7.0, 10.0]
 [8.0, 0.0]   [8.0, 1.0]   [8.0, 2.0]      [8.0, 9.0]   [8.0, 10.0]
 [9.0, 0.0]   [9.0, 1.0]   [9.0, 2.0]      [9.0, 9.0]   [9.0, 10.0]
 [10.0, 0.0]  [10.0, 1.0]  [10.0, 2.0]  …  [10.0, 9.0]  [10.0, 10.0]

julia> Marble.whichblock(lattice, Vec(8.5, 1.5))
CartesianIndex(2, 1)
```
"""
@inline function whichblock(lattice::Lattice, x::Vec)
    I = whichcell(lattice, x)
    I === nothing && return nothing
    CartesianIndex(@. (I.I-1) >> BLOCK_UNIT + 1)
end

blocksize(gridsize::Tuple{Vararg{Int}}) = (ncells = gridsize .- 1; @. (ncells - 1) >> BLOCK_UNIT + 1)

function threadsafe_blocks(blocksize::NTuple{dim, Int}) where {dim}
    starts = AxisArray(nfill([1,2], Val(dim)))
    vec(map(st -> map(CartesianIndex{dim}, AxisArray(StepRange.(st, 2, blocksize))), starts))
end
