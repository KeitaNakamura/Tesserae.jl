abstract type Interpolation end
abstract type Kernel <: Interpolation end

function create_property(::Type{Vec{dim, T}}, it::Interpolation) where {dim, T}
    dims = nfill(gridspan(it), Val(dim))
    N = zeros(T, dims)
    ∇N = zeros(Vec{dim, T}, dims)
    (; N, ∇N)
end

"""
    MPValues(Vec{dim}, interpolation)
    MPValues(Vec{dim, T}, interpolation)

`MPValues` stores properties for interpolation, such as the value of the kernel and its gradient.

```jldoctest
julia> lattice = Lattice(1.0, (0,5), (0,5)); # computational domain

julia> x = Vec(2.2, 3.4); # particle coordinate

julia> mp = MPValues(Vec{2}, QuadraticBSpline())
MPValues:
  Interpolation: QuadraticBSpline()
  Property names: N::Matrix{Float64}, ∇N::Matrix{Vec{2, Float64}}
  Neighbor nodes: CartesianIndices((0:0, 0:0))

julia> update!(mp, x, lattice) # update `mp` at position `x` in `lattice`
MPValues:
  Interpolation: QuadraticBSpline()
  Property names: N::Matrix{Float64}, ∇N::Matrix{Vec{2, Float64}}
  Neighbor nodes: CartesianIndices((2:4, 3:5))

julia> sum(mp.N)
1.0000000000000004

julia> sum(mp.∇N)
2-element Vec{2, Float64}:
 0.0
 5.551115123125783e-17

julia> neighbornodes(mp) # grid indices within the local domain of a particle
CartesianIndices((2:4, 3:5))
```
"""
struct MPValues{It, Prop <: NamedTuple, Indices <: AbstractArray{<: Any, 0}}
    it::It
    prop::Prop
    indices::Indices
end

function MPValues(::Type{Vec{dim, T}}, it::Interpolation) where {dim, T}
    prop = create_property(Vec{dim, T}, it)
    indices = CartesianIndices(nfill(0:0, Val(dim)))
    MPValues(it, prop, fill(indices))
end
MPValues(::Type{Vec{dim}}, it::Interpolation) where {dim} = MPValues(Vec{dim, Float64}, it)

Base.propertynames(mp::MPValues) = propertynames(getfield(mp, :prop))
@inline function Base.getproperty(mp::MPValues, name::Symbol)
    getproperty(getfield(mp, :prop), name)
end
@inline function Base.setproperty!(mp::MPValues, name::Symbol, v)
    setproperty!(getfield(mp, :prop), name, v)
end

@inline interpolation(mp::MPValues) = getfield(mp, :it)

@inline neighbornodes(mp::MPValues) = getfield(mp, :indices)[]
@inline function neighbornodes(mp::MPValues, grid::Grid)
    inds = neighbornodes(mp)
    @boundscheck checkbounds(grid, inds)
    inds
end
@inline function neighbornodes(mp::MPValues, grid::SpGrid)
    inds = neighbornodes(mp)
    spinds = get_spinds(grid)
    @boundscheck checkbounds(spinds, inds)
    @inbounds neighbors = view(spinds, inds)
    @debug @assert all(isactive, neighbors)
    neighbors
end

@inline function set_neighbornodes!(mp::MPValues, indices)
    getfield(mp, :indices)[] = indices
    mp
end

function Base.show(io::IO, mp::MPValues)
    print(io, "MPValues: \n")
    print(io, "  Interpolation: ", interpolation(mp), "\n")
    print(io, "  Property names: ")
    print(io, join(map(propertynames(mp)) do name
        string(name, "::", typeof(getproperty(mp, name)))
    end, ", "), "\n")
    print(io, "  Neighbor nodes: ", neighbornodes(mp))
end

struct MPValuesVector{It, Prop <: NamedTuple, Indices, ElType <: MPValues{It}} <: AbstractVector{ElType}
    it::It
    prop::Prop
    indices::Indices
end

function MPValuesVector(::Type{Vec{dim, T}}, it::Interpolation, n::Int) where {dim, T}
    prop = map(create_property(Vec{dim, T}, it)) do prop
        fill(zero(eltype(prop)), size(prop)..., n)
    end
    indices = fill(CartesianIndices(nfill(0:0, Val(dim))), n)
    It = typeof(it)
    Prop = typeof(prop)
    Indices = typeof(indices)
    ElType = Base._return_type(_getindex, Tuple{It, Prop, Indices, Int})
    MPValuesVector{It, Prop, Indices, ElType}(it, prop, indices)
end
MPValuesVector(::Type{Vec{dim}}, it::Interpolation, n::Int) where {dim} = MPValuesVector(Vec{dim, Float64}, it, n)

Base.IndexStyle(::Type{<: MPValuesVector}) = IndexLinear()
Base.size(x::MPValuesVector) = size(x.indices)

@inline interpolation(mp::MPValuesVector) = getfield(mp, :it)

@inline function Base.getindex(x::MPValuesVector, i::Integer)
    @boundscheck checkbounds(x, i)
    @inbounds _getindex(x.it, x.prop, x.indices, i)
end
@generated function _getindex(it::Interpolation, prop::NamedTuple{names}, indices, i::Integer) where {names}
    exps = [:(viewcol(prop.$name, i)) for name in names]
    quote
        @_inline_meta
        @_propagate_inbounds_meta
        MPValues(it, NamedTuple{names}(tuple($(exps...))), view(indices, i))
    end
end

@inline function viewcol(A::AbstractArray, i::Integer)
    @boundscheck checkbounds(axes(A, ndims(A)), i)
    colons = nfill(:, Val(ndims(A)-1))
    @inbounds view(A, colons..., i)
end

function Base.show(io::IO, mime::MIME"text/plain", mpvalues::MPValuesVector)
    print(io, length(mpvalues), "-element MPValuesVector: \n")
    print(io, "  Interpolation: ", interpolation(mpvalues))
end

function Base.show(io::IO, mpvalues::MPValuesVector)
    print(io, length(mpvalues), "-element MPValuesVector: \n")
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

function update!(mp::MPValues, pt, lattice)
    set_neighbornodes!(mp, neighbornodes(interpolation(mp), pt, lattice))
    update_property!(mp, pt, lattice)
    mp
end

function update!(mp::MPValues, pt, lattice, filter)
    @debug @assert size(lattice) == size(filter)
    set_neighbornodes!(mp, neighbornodes(interpolation(mp), pt, lattice))
    update_property!(mp, pt, lattice, filter)
    mp
end
