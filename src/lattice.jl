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

"""
    Lattice(Δx, (xmin, xmax), (ymin, ymax)...)
    Lattice(T, Δx, (xmin, xmax), (ymin, ymax)...)

Construct `Lattice` with the spacing `Δx`.

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
Base.IndexStyle(::Type{<: Lattice}) = IndexCartesian()
# helpers
spacing(x::Lattice) = x.dx
spacing_inv(x::Lattice) = x.dx_inv
get_axes(x::Lattice) = get_axes(x.axisarray)
get_axes(x::Lattice, i::Integer) = (@_propagate_inbounds_meta; get_axes(x)[i])

function Lattice(axes::Vararg{AbstractRange, dim}) where {dim}
    @assert allequal(map(step, axes))
    dx = step(first(axes))
    Lattice(AxisArray(axes), dx, inv(dx))
end

function Lattice(::Type{T}, dx::Real, minmax::Vararg{Tuple{Real, Real}, dim}) where {T, dim}
    @assert all(map(issorted, minmax))
    axisarray = AxisArray(map(lims->Vector{T}(range(lims...; step=dx)), minmax))
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

"""
    isinside(x::Vec, lattice::Lattice)

Check if `x` is inside the `lattice`.
This returns `true` if `all(lattice[1] .≤ x .< lattice[end])`.
"""
@inline function isinside(x::Vec{dim}, lattice::Lattice{dim}) where {dim}
    xmin = lattice[1]
    dx⁻¹ = spacing_inv(lattice)
    ξ = Tuple((x - xmin) * dx⁻¹)
    isinside(ξ, size(lattice))
end
@generated function isinside(ξ::NTuple{dim}, dims::Dims{dim}) where {dim}
    quote
        @_inline_meta
        @nall $dim d -> 0 ≤ ξ[d] < dims[d]-1
    end
end

"""
    neighbornodes(x::Vec, h::Real, lattice::Lattice)

Return `CartesianIndices` storing neighboring node around `x`.
`h` denotes the range for searching area. In 1D, for example, the range `a`
becomes ` x-h*Δx ≤ a < x+h*Δx` where `Δx` is `spacing(lattice)`.

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

julia> neighbornodes(Vec(1.5), 1, lattice)
CartesianIndices((2:3,))

julia> neighbornodes(Vec(1.5), 3, lattice)
CartesianIndices((1:5,))
```
"""
@inline function neighbornodes(x::Vec, h::Real, lattice::Lattice{dim, T}) where {dim, T}
    xmin = lattice[1]
    dx⁻¹ = spacing_inv(lattice)
    dims = size(lattice)
    ξ = Tuple((x - xmin) * dx⁻¹)
    isinside(ξ, dims) || return CartesianIndices(nfill(0:0, Val(dim)))
    start = @. unsafe_trunc(Int, floor(ξ - h)) + 2
    stop  = @. unsafe_trunc(Int, floor(ξ + h)) + 1
    imin = Tuple(@. max(start, 1))
    imax = Tuple(@. min(stop, dims))
    CartesianIndices(UnitRange.(imin, imax))
end

"""
    whichcell(x::Vec, lattice::Lattice)

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

julia> whichcell(Vec(1.5, 1.5), lattice)
CartesianIndex(2, 2)
```
"""
@inline function whichcell(x::Vec, lattice::Lattice{dim, T}) where {dim, T}
    xmin = lattice[1]
    dx⁻¹ = spacing_inv(lattice)
    ξ = Tuple((x - xmin) * dx⁻¹)
    isinside(ξ, size(lattice)) || return nothing
    CartesianIndex(Tuple(@. unsafe_trunc(Int, floor(ξ)) + 1))
end
