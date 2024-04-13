abstract type Interpolation end
abstract type Kernel <: Interpolation end

function create_property(::Type{Vec{dim, T}}, it::Interpolation) where {dim, T}
    dims = nfill(gridspan(it), Val(dim))
    N = zeros(T, dims)
    ∇N = zeros(Vec{dim, T}, dims)
    (; N, ∇N)
end

"""
    MPValue(Vec{dim}, interpolation)
    MPValue(Vec{dim, T}, interpolation)

`MPValue` stores properties for interpolation, such as the value of the kernel and its gradient.

```jldoctest
julia> mesh = CartesianMesh(1.0, (0,5), (0,5)); # computational domain

julia> x = Vec(2.2, 3.4); # particle coordinate

julia> mp = MPValue(Vec{2}, QuadraticBSpline())
MPValue:
  Interpolation: QuadraticBSpline()
  Property names: N::Matrix{Float64}, ∇N::Matrix{Vec{2, Float64}}
  Neighboring nodes: CartesianIndices((0:0, 0:0))

julia> update!(mp, x, mesh) # update `mp` at position `x` in `mesh`
MPValue:
  Interpolation: QuadraticBSpline()
  Property names: N::Matrix{Float64}, ∇N::Matrix{Vec{2, Float64}}
  Neighboring nodes: CartesianIndices((2:4, 3:5))

julia> sum(mp.N)
1.0000000000000004

julia> sum(mp.∇N)
2-element Vec{2, Float64}:
 0.0
 5.551115123125783e-17

julia> neighboringnodes(mp) # grid indices within the local domain of a particle
CartesianIndices((2:4, 3:5))
```
"""
struct MPValue{It, Prop <: NamedTuple, Indices <: AbstractArray{<: Any, 0}}
    it::It
    prop::Prop
    indices::Indices
end

function MPValue(::Type{Vec{dim, T}}, it::Interpolation) where {dim, T}
    prop = create_property(Vec{dim, T}, it)
    indices = ZeroCartesianIndices(Val(dim))
    MPValue(it, prop, fill(indices))
end
MPValue(::Type{Vec{dim}}, it::Interpolation) where {dim} = MPValue(Vec{dim, Float64}, it)

Base.propertynames(mp::MPValue) = propertynames(getfield(mp, :prop))
@inline function Base.getproperty(mp::MPValue, name::Symbol)
    getproperty(getfield(mp, :prop), name)
end
@inline function Base.setproperty!(mp::MPValue, name::Symbol, v)
    setproperty!(getfield(mp, :prop), name, v)
end

@inline interpolation(mp::MPValue) = getfield(mp, :it)

@inline neighboringnodes(mp::MPValue) = getfield(mp, :indices)[]
@inline function neighboringnodes(mp::MPValue, grid::Grid)
    inds = neighboringnodes(mp)
    @boundscheck checkbounds(grid, inds)
    inds
end
@inline function neighboringnodes(mp::MPValue, grid::SpGrid)
    inds = neighboringnodes(mp)
    spinds = get_spinds(grid)
    @boundscheck checkbounds(spinds, inds)
    @inbounds neighbors = view(spinds, inds)
    @debug @assert all(isactive, neighbors)
    neighbors
end

@inline function set_neighboringnodes!(mp::MPValue, indices)
    getfield(mp, :indices)[] = indices
    mp
end

function Base.show(io::IO, mp::MPValue)
    print(io, "MPValue: \n")
    print(io, "  Interpolation: ", interpolation(mp), "\n")
    print(io, "  Property names: ")
    print(io, join(map(propertynames(mp)) do name
        string(name, "::", typeof(getproperty(mp, name)))
    end, ", "), "\n")
    print(io, "  Neighboring nodes: ", neighboringnodes(mp))
end

struct MPValueVector{It, Prop <: NamedTuple, Indices, ElType <: MPValue{It}} <: AbstractVector{ElType}
    it::It
    prop::Prop
    indices::Indices
end

function MPValueVector(::Type{Vec{dim, T}}, it::Interpolation, n::Int) where {dim, T}
    prop = map(create_property(Vec{dim, T}, it)) do prop
        fill(zero(eltype(prop)), size(prop)..., n)
    end
    indices = fill(ZeroCartesianIndices(Val(dim)), n)
    It = typeof(it)
    Prop = typeof(prop)
    Indices = typeof(indices)
    ElType = Base._return_type(_getindex, Tuple{It, Prop, Indices, Int})
    MPValueVector{It, Prop, Indices, ElType}(it, prop, indices)
end
MPValueVector(::Type{Vec{dim}}, it::Interpolation, n::Int) where {dim} = MPValueVector(Vec{dim, Float64}, it, n)

Base.IndexStyle(::Type{<: MPValueVector}) = IndexLinear()
Base.size(x::MPValueVector) = size(x.indices)

@inline interpolation(mp::MPValueVector) = getfield(mp, :it)

@inline function Base.getindex(x::MPValueVector, i::Integer)
    @boundscheck checkbounds(x, i)
    @inbounds _getindex(x.it, x.prop, x.indices, i)
end
@generated function _getindex(it::Interpolation, prop::NamedTuple{names}, indices, i::Integer) where {names}
    exps = [:(viewcol(prop.$name, i)) for name in names]
    quote
        @_inline_meta
        @_propagate_inbounds_meta
        MPValue(it, NamedTuple{names}(tuple($(exps...))), view(indices, i))
    end
end

@inline function viewcol(A::AbstractArray, i::Integer)
    @boundscheck checkbounds(axes(A, ndims(A)), i)
    colons = nfill(:, Val(ndims(A)-1))
    @inbounds view(A, colons..., i)
end

function Base.show(io::IO, mime::MIME"text/plain", mpvalues::MPValueVector)
    print(io, length(mpvalues), "-element MPValueVector: \n")
    print(io, "  Interpolation: ", interpolation(mpvalues))
end

function Base.show(io::IO, mpvalues::MPValueVector)
    print(io, length(mpvalues), "-element MPValueVector: \n")
    print(io, "  Interpolation: ", interpolation(mpvalues))
end

###########
# update! #
###########

@inline function alltrue(A::AbstractArray{Bool}, indices::CartesianIndices)
    @debug checkbounds(A, indices)
    @inbounds @simd for i in indices
        A[i] || return false
    end
    true
end
@inline function alltrue(A::Trues, indices::CartesianIndices)
    @debug checkbounds(A, indices)
    true
end

function update!(mp::MPValue, pt, mesh)
    set_neighboringnodes!(mp, neighboringnodes(interpolation(mp), pt, mesh))
    update_property!(mp, pt, mesh)
    mp
end

function update!(mp::MPValue, pt, mesh, filter)
    @debug @assert size(mesh) == size(filter)
    set_neighboringnodes!(mp, neighboringnodes(interpolation(mp), pt, mesh))
    update_property!(mp, pt, mesh, filter)
    mp
end
