abstract type AbstractMesh{dim, T, N} <: AbstractArray{Vec{dim, T}, N} end

fillzero!(x::AbstractMesh) = x

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

Base.copy(A::AxisArray) = AxisArray(map(copy, A.axes))

"""
    CartesianMesh(h, (xmin, xmax), (ymin, ymax)...)
    CartesianMesh(T, h, (xmin, xmax), (ymin, ymax)...)

Construct `CartesianMesh` with the spacing `h`.

# Examples
```jldoctest
julia> CartesianMesh(1.0, (0,3), (1,4))
4×4 CartesianMesh{2, Float64, Vector{Float64}}:
 [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]
 [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]
 [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]
 [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]
```
"""
struct CartesianMesh{dim, T, V <: AbstractVector{T}} <: AbstractMesh{dim, T, dim}
    axisarray::AxisArray{dim, T, V}
    h::T
    h_inv::T
end

get_axisarray(x::CartesianMesh) = x.axisarray
Base.size(x::CartesianMesh) = size(get_axisarray(x))
Base.IndexStyle(::Type{<: CartesianMesh}) = IndexCartesian()
# helpers
spacing(x::CartesianMesh) = x.h
spacing_inv(x::CartesianMesh) = x.h_inv
get_axes(x::CartesianMesh) = get_axes(x.axisarray)
get_axes(x::CartesianMesh, i::Integer) = (@_propagate_inbounds_meta; get_axes(x)[i])
@inline get_xmin(x::CartesianMesh{dim}) where {dim} = @inbounds x[oneunit(CartesianIndex{dim})]
@inline get_xmax(x::CartesianMesh{dim}) where {dim} = @inbounds x[size(x)...]
volume(x::CartesianMesh) = prod(get_xmax(x) - get_xmin(x))

function CartesianMesh(axes::Vararg{AbstractRange, dim}) where {dim}
    @assert all(ax->step(ax)==step(first(axes)), axes)
    h = step(first(axes))
    CartesianMesh(AxisArray(axes), h, inv(h))
end

function CartesianMesh(::Type{T}, h::Real, minmax::Vararg{Tuple{Real, Real}, dim}) where {T, dim}
    @assert all(map(issorted, minmax))
    axisarray = AxisArray(map(lims->Vector{T}(range(lims...; step=h)), minmax))
    CartesianMesh(axisarray, T(h), T(inv(h)))
end
CartesianMesh(h::Real, minmax::Tuple{Real, Real}...) = CartesianMesh(Float64, h, minmax...)

@inline function Base.getindex(mesh::CartesianMesh{dim}, i::Vararg{Integer, dim}) where {dim}
    @boundscheck checkbounds(mesh, i...)
    @inbounds Vec(get_axisarray(mesh)[i...])
end
@inline function Base.getindex(mesh::CartesianMesh{dim}, ranges::Vararg{AbstractUnitRange{Int}, dim}) where {dim}
    @boundscheck checkbounds(mesh, ranges...)
    @inbounds CartesianMesh(get_axisarray(mesh)[ranges...], spacing(mesh), spacing_inv(mesh))
end

Base.copy(mesh::CartesianMesh) = CartesianMesh(copy(get_axisarray(mesh)), spacing(mesh), spacing_inv(mesh))

# normalize `x` by `mesh`
@inline function Tensorial.normalize(x::Vec{dim}, mesh::CartesianMesh{dim}) where {dim}
    xmin = get_xmin(mesh)
    h⁻¹ = spacing_inv(mesh)
    (x - xmin) * h⁻¹
end

"""
    isinside(x::Vec, mesh::CartesianMesh)

Check if `x` is inside the `mesh`.
This returns `true` if `all(mesh[1] .≤ x .< mesh[end])`.
"""
@inline function isinside(x::Vec{dim}, mesh::CartesianMesh{dim}) where {dim}
    ξ = Tuple(normalize(x, mesh))
    isinside(ξ, size(mesh))
end
@inline function isinside(ξ::NTuple{dim}, dims::Dims{dim}) where {dim}
    !isnothing(whichcell(ξ, dims))
end

"""
    neighboringnodes(x::Vec, r::Real, mesh::CartesianMesh)

Return `CartesianIndices` for neighboring nodes around `x`.
`r` denotes the range for searching area. In 1D, for example, the range `a`
becomes ` x-r*h ≤ a < x+r*h` where `h` is `spacing(mesh)`.

# Examples
```jldoctest
julia> mesh = CartesianMesh(1, (0,5))
6-element CartesianMesh{1, Float64, Vector{Float64}}:
 [0.0]
 [1.0]
 [2.0]
 [3.0]
 [4.0]
 [5.0]

julia> neighboringnodes(Vec(1.5), 1, mesh)
CartesianIndices((2:3,))

julia> neighboringnodes(Vec(1.5), 3, mesh)
CartesianIndices((1:5,))
```
"""
@inline function neighboringnodes(x::Vec, r::Real, mesh::CartesianMesh{dim, T}) where {dim, T}
    ξ = Tuple(normalize(x, mesh))
    dims = size(mesh)
    isinside(ξ, dims) || return EmptyCartesianIndices(Val(dim))
    start = @. unsafe_trunc(Int, floor(ξ - r)) + 2
    stop  = @. unsafe_trunc(Int, floor(ξ + r)) + 1
    imin = Tuple(@. max(start, 1))
    imax = Tuple(@. min(stop, dims))
    CartesianIndices(UnitRange.(imin, imax))
end
@inline EmptyCartesianIndices(::Val{dim}) where {dim} = CartesianIndices(nfill(1:0, Val(dim)))

"""
    whichcell(x::Vec, mesh::CartesianMesh)

Return cell index where `x` locates.

# Examples
```jldoctest
julia> mesh = CartesianMesh(1, (0,5), (0,5))
6×6 CartesianMesh{2, Float64, Vector{Float64}}:
 [0.0, 0.0]  [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]  [0.0, 5.0]
 [1.0, 0.0]  [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]  [1.0, 5.0]
 [2.0, 0.0]  [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]  [2.0, 5.0]
 [3.0, 0.0]  [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]  [3.0, 5.0]
 [4.0, 0.0]  [4.0, 1.0]  [4.0, 2.0]  [4.0, 3.0]  [4.0, 4.0]  [4.0, 5.0]
 [5.0, 0.0]  [5.0, 1.0]  [5.0, 2.0]  [5.0, 3.0]  [5.0, 4.0]  [5.0, 5.0]

julia> whichcell(Vec(1.5, 1.5), mesh)
CartesianIndex(2, 2)
```
"""
@inline function whichcell(x::Vec, mesh::CartesianMesh{dim, T}) where {dim, T}
    ξ = Tuple(normalize(x, mesh))
    whichcell(ξ, size(mesh))
end

@generated function whichcell(ξ::NTuple{dim, T}, gridsize::Dims{dim}) where {dim, T}
    quote
        @_inline_meta
        index = map(floor, ξ)
        isinside = @nall $dim d -> T(0) ≤ index[d] ≤ T(gridsize[d]-2)
        ifelse(isinside, CartesianIndex(@. unsafe_trunc(Int, index) + 1), nothing)
    end
end
