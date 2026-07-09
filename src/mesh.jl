abstract type AbstractMesh{dim, T, N} <: AbstractArray{Vec{dim, T}, N} end

fillzero!(x::AbstractMesh) = x

const BLOCK_SIZE_LOG2 = Int(Preferences.@load_preference("block_size_log2", 2)) # default 2^n cells per block

function _check_block_size_log2(::Val{L}) where {L}
    L isa Integer || throw(ArgumentError("block_size_log2 must be an integer Val, got Val{$L}()"))
    L < 0 && throw(ArgumentError("block_size_log2 must be non-negative, got $L"))
    nothing
end

"""
    CartesianMesh([T,] h, (xmin, xmax), (ymin, ymax)...; warn=true, block_size_log2=Val(2))

Construct a uniform Cartesian mesh with scalar spacing `h` (same in all directions).
If an axis length is not divisible by `h`, the upper bound is expanded to cover the
requested domain. Set `warn=false` to suppress the expansion warning.
`block_size_log2` sets the block decomposition used by [`ThreadPartition`](@ref) and [`SpArray`](@ref)
grids generated from this mesh.

# Examples
```jldoctest
julia> CartesianMesh(1.0, (0,3), (1,4))
4×4 CartesianMesh{2, Float64, Vector{Float64}, 2}:
 [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]
 [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]
 [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]
 [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]
```
"""
struct CartesianMesh{dim, T, V <: AbstractVector{T}, L} <: AbstractMesh{dim, T, dim}
    axes::NTuple{dim, V}
    h::T
    h_inv::T
end

Base.size(mesh::CartesianMesh) = map(length, mesh.axes)
Base.IndexStyle(::Type{<: CartesianMesh}) = IndexCartesian()

"""
    spacing(::CartesianMesh)

Return the spacing of the mesh.
"""
spacing(mesh::CartesianMesh) = mesh.h
spacing_inv(mesh::CartesianMesh) = mesh.h_inv
block_size_log2(::CartesianMesh{dim, T, V, L}) where {dim, T, V, L} = L

@inline get_xmin(x::CartesianMesh{dim}) where {dim} = @inbounds x[oneunit(CartesianIndex{dim})]
@inline get_xmax(x::CartesianMesh{dim}) where {dim} = @inbounds x[size(x)...]

"""
    volume(::CartesianMesh)

Return the volume of the mesh.
"""
volume(mesh::CartesianMesh) = prod(get_xmax(mesh) - get_xmin(mesh))

function CartesianMesh(axes::NTuple{dim, V}, h::Real, h_inv::Real; block_size_log2::Val{L}=Val(BLOCK_SIZE_LOG2)) where {dim, T, V <: AbstractVector{T}, L}
    _check_block_size_log2(block_size_log2)
    CartesianMesh{dim, T, V, L}(axes, T(h), T(h_inv))
end

function CartesianMesh(axes::Vararg{AbstractRange{<: AbstractFloat}, dim}; block_size_log2::Val{L}=Val(BLOCK_SIZE_LOG2)) where {dim, L}
    @assert all(ax->step(ax)==step(first(axes)), axes)
    h = step(first(axes))
    CartesianMesh(axes, h, inv(h); block_size_log2)
end

function _covered_axis(h::Real, lims::Tuple{Real, Real}; warn::Bool)
    xmin, xmax = lims
    ax = range(xmin, xmax; step=h)
    n = length(ax)
    if !isapprox(last(ax), xmax)
        actual_xmax = xmin + n*h
        n += 1
        warn && @warn "CartesianMesh axis length is not divisible by spacing; expanding the upper bound to cover the requested domain" requested=lims actual=(xmin, actual_xmax) spacing=h
    end
    range(xmin; step=h, length=n)
end

function CartesianMesh(::Type{T}, h::Real, minmax::Vararg{Tuple{Real, Real}, dim}; pad::Int = 0, warn::Bool = true, block_size_log2::Val{L}=Val(BLOCK_SIZE_LOG2)) where {T, dim, L}
    @assert all(x->x[1]<x[2], minmax)

    axes = map(minmax) do lims
        ax = _covered_axis(h, lims; warn)
        pad == 0 && return Vector{T}(ax)

        start = first(ax) - pad*h
        n  = length(ax) + 2*pad
        Vector{T}(range(start; step=h, length=n))
    end

    CartesianMesh(axes, T(h), T(inv(h)); block_size_log2)
end
CartesianMesh(h::Real, minmax::Tuple{Real, Real}...; kwargs...) = CartesianMesh(Float64, h, minmax...; kwargs...)

"""
    extract(mesh::CartesianMesh, (xmin, xmax), (ymin, ymax)...)

Extract a portion of the `mesh`.
The extracted mesh retains the original origin and spacing.
"""
function extract(mesh::CartesianMesh{dim}, minmax::Vararg{Tuple{Real, Real}, dim}) where {dim}
    @assert all(x->x[1]<x[2], minmax)
    indices = CartesianIndices(ntuple(Val(dim)) do d
        ax = mesh.axes[d]
        xmin, xmax = minmax[d]
        imin =  findlast(x -> (x < xmin) || (x ≈ xmin), ax)
        imax = findfirst(x -> (x > xmax) || (x ≈ xmax), ax)
        imin:imax
    end)
    mesh[indices]
end

axismesh(mesh::CartesianMesh, d::Integer) = CartesianMesh((mesh.axes[d],), spacing(mesh), spacing_inv(mesh); block_size_log2=Val(block_size_log2(mesh)))

@generated function Base.getindex(mesh::CartesianMesh{dim}, I::Vararg{Integer, dim}) where {dim}
    quote
        @_inline_meta
        @boundscheck checkbounds(mesh, I...)
        @inbounds Vec(@ntuple $dim d -> mesh.axes[d][I[d]])
    end
end
@generated function Base.getindex(mesh::CartesianMesh{dim}, ranges::Vararg{AbstractUnitRange{Int}, dim}) where {dim}
    quote
        @_inline_meta
        @boundscheck checkbounds(mesh, ranges...)
        @inbounds CartesianMesh((@ntuple $dim d -> mesh.axes[d][ranges[d]]), spacing(mesh), spacing_inv(mesh); block_size_log2=Val(block_size_log2(mesh)))
    end
end
@inline function Base.getindex(mesh::CartesianMesh, indices::CartesianIndices)
    @_propagate_inbounds_meta
    mesh[indices.indices...]
end

Base.copy(mesh::CartesianMesh) = CartesianMesh(copy(get_axisarray(mesh)), spacing(mesh), spacing_inv(mesh); block_size_log2=Val(block_size_log2(mesh)))

# normalize `x` by `mesh`
@inline function Tensorial.normalize(x::Vec{dim}, mesh::CartesianMesh{dim}) where {dim}
    xmin = get_xmin(mesh)
    h⁻¹ = spacing_inv(mesh)
    (x - xmin) * h⁻¹
end

"""
    isinside(x::Vec, mesh::CartesianMesh)

Check if `x` is inside the `mesh`.
"""
@inline function isinside(x::Vec{dim}, mesh::CartesianMesh{dim}) where {dim}
    ξ = Tuple(normalize(x, mesh))
    isinside(ξ, size(mesh))
end
@inline function isinside(ξ::NTuple{dim}, dims::Dims{dim}) where {dim}
    !isnothing(_findcell(ξ, dims))
end

"""
    supportnodes(x::Vec, r::Real, mesh::CartesianMesh)

Return `CartesianIndices` for support nodes around `x`.
`r` denotes the range for searching area. In 1D, for example, the range `a`
becomes ` x-r*h ≤ a < x+r*h` where `h` is `spacing(mesh)`.

# Examples
```jldoctest
julia> mesh = CartesianMesh(1, (0,5))
6-element CartesianMesh{1, Float64, Vector{Float64}, 2}:
 [0.0]
 [1.0]
 [2.0]
 [3.0]
 [4.0]
 [5.0]

julia> supportnodes(Vec(1.5), 1, mesh)
CartesianIndices((2:3,))

julia> supportnodes(Vec(1.5), 3, mesh)
CartesianIndices((1:5,))
```
"""
@inline function supportnodes(x::Vec, r::Real, mesh::CartesianMesh{dim, T}) where {dim, T}
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
    findcell(x::Vec, mesh::CartesianMesh)

Return the cell index where `x` is located.

# Examples
```jldoctest
julia> mesh = CartesianMesh(1, (0,5), (0,5))
6×6 CartesianMesh{2, Float64, Vector{Float64}, 2}:
 [0.0, 0.0]  [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]  [0.0, 5.0]
 [1.0, 0.0]  [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]  [1.0, 5.0]
 [2.0, 0.0]  [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]  [2.0, 5.0]
 [3.0, 0.0]  [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]  [3.0, 5.0]
 [4.0, 0.0]  [4.0, 1.0]  [4.0, 2.0]  [4.0, 3.0]  [4.0, 4.0]  [4.0, 5.0]
 [5.0, 0.0]  [5.0, 1.0]  [5.0, 2.0]  [5.0, 3.0]  [5.0, 4.0]  [5.0, 5.0]

julia> findcell(Vec(1.5, 1.5), mesh)
CartesianIndex(2, 2)
```
"""
@inline function findcell(x::Vec, mesh::CartesianMesh{dim, T}) where {dim, T}
    ξ = Tuple(normalize(x, mesh))
    _findcell(ξ, size(mesh))
end

@generated function _findcell(ξ::NTuple{dim, T}, gridsize::Dims{dim}) where {dim, T}
    quote
        @_inline_meta
        index = map(floor, ξ)
        isinside = @nall $dim d -> T(0) ≤ index[d] ≤ T(gridsize[d]-2)
        ifelse(isinside, CartesianIndex(@. unsafe_trunc(Int, index) + 1), nothing)
    end
end
