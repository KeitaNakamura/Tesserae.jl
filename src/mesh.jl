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

"""
    CartesianMesh(Δx, (xmin, xmax), (ymin, ymax)...)
    CartesianMesh(T, Δx, (xmin, xmax), (ymin, ymax)...)

Construct `CartesianMesh` with the spacing `Δx`.

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
    dx::T
    dx_inv::T
end

get_axisarray(x::CartesianMesh) = x.axisarray
Base.size(x::CartesianMesh) = size(get_axisarray(x))
Base.IndexStyle(::Type{<: CartesianMesh}) = IndexCartesian()
# helpers
spacing(x::CartesianMesh) = x.dx
spacing_inv(x::CartesianMesh) = x.dx_inv
get_axes(x::CartesianMesh) = get_axes(x.axisarray)
get_axes(x::CartesianMesh, i::Integer) = (@_propagate_inbounds_meta; get_axes(x)[i])

function CartesianMesh(axes::Vararg{AbstractRange, dim}) where {dim}
    @assert all(ax->step(ax)==step(first(axes)), axes)
    dx = step(first(axes))
    CartesianMesh(AxisArray(axes), dx, inv(dx))
end

function CartesianMesh(::Type{T}, dx::Real, minmax::Vararg{Tuple{Real, Real}, dim}) where {T, dim}
    @assert all(map(issorted, minmax))
    axisarray = AxisArray(map(lims->Vector{T}(range(lims...; step=dx)), minmax))
    CartesianMesh(axisarray, T(dx), T(inv(dx)))
end
CartesianMesh(dx::Real, minmax::Tuple{Real, Real}...) = CartesianMesh(Float64, dx, minmax...)

@inline function Base.getindex(mesh::CartesianMesh{dim}, i::Vararg{Integer, dim}) where {dim}
    @boundscheck checkbounds(mesh, i...)
    @inbounds Vec(get_axisarray(mesh)[i...])
end
@inline function Base.getindex(mesh::CartesianMesh{dim}, ranges::Vararg{AbstractUnitRange{Int}, dim}) where {dim}
    @boundscheck checkbounds(mesh, ranges...)
    @inbounds CartesianMesh(get_axisarray(mesh)[ranges...], spacing(mesh), spacing_inv(mesh))
end

"""
    isinside(x::Vec, mesh::CartesianMesh)

Check if `x` is inside the `mesh`.
This returns `true` if `all(mesh[1] .≤ x .< mesh[end])`.
"""
@inline function isinside(x::Vec{dim}, mesh::CartesianMesh{dim}) where {dim}
    xmin = mesh[1]
    dx⁻¹ = spacing_inv(mesh)
    ξ = Tuple((x - xmin) * dx⁻¹)
    isinside(ξ, size(mesh))
end
@generated function isinside(ξ::NTuple{dim}, dims::Dims{dim}) where {dim}
    quote
        @_inline_meta
        @nall $dim d -> 0 ≤ ξ[d] < dims[d]-1
    end
end

"""
    neighboringnodes(x::Vec, h::Real, mesh::CartesianMesh)

Return `CartesianIndices` for neighboring nodes around `x`.
`h` denotes the range for searching area. In 1D, for example, the range `a`
becomes ` x-h*Δx ≤ a < x+h*Δx` where `Δx` is `spacing(mesh)`.

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
@inline function neighboringnodes(x::Vec, h::Real, mesh::CartesianMesh{dim, T}) where {dim, T}
    xmin = mesh[1]
    dx⁻¹ = spacing_inv(mesh)
    dims = size(mesh)
    ξ = Tuple((x - xmin) * dx⁻¹)
    isinside(ξ, dims) || return CartesianIndices(nfill(0:0, Val(dim)))
    start = @. unsafe_trunc(Int, floor(ξ - h)) + 2
    stop  = @. unsafe_trunc(Int, floor(ξ + h)) + 1
    imin = Tuple(@. max(start, 1))
    imax = Tuple(@. min(stop, dims))
    CartesianIndices(UnitRange.(imin, imax))
end

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
    xmin = mesh[1]
    dx⁻¹ = spacing_inv(mesh)
    ξ = Tuple((x - xmin) * dx⁻¹)
    isinside(ξ, size(mesh)) || return nothing
    CartesianIndex(Tuple(@. unsafe_trunc(Int, floor(ξ)) + 1))
end
