abstract type Interpolation end
abstract type Kernel <: Interpolation end

function create_property(::Type{Vec{dim, T}}, it::Interpolation, ::Val{diff}) where {dim, T, diff}
    dims = nfill(gridspan(it), Val(dim))
    diff === 0 && return (; N=zeros(T, dims))
    diff === 1 && return (; N=zeros(T, dims), ∇N=zeros(Vec{dim, T}, dims))
    diff === 2 && return (; N=zeros(T, dims), ∇N=zeros(Vec{dim, T}, dims), ∇∇N=zeros(SymmetricSecondOrderTensor{dim, T}, dims))
    error("wrong differentiation order, choose `0`, `1` or `2`")
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
  Neighboring nodes: CartesianIndices((1:0, 1:0))

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

function MPValue(::Type{Vec{dim, T}}, it::Interpolation, diff::Val=Val(1)) where {dim, T}
    prop = create_property(Vec{dim, T}, it, diff)
    indices = fill(EmptyCartesianIndices(Val(dim)))
    MPValue(it, prop, indices)
end
MPValue(::Type{Vec{dim}}, it::Interpolation, diff::Val=Val(1)) where {dim} = MPValue(Vec{dim, Float64}, it, diff)

Base.propertynames(mp::MPValue) = propertynames(getfield(mp, :prop))
@inline function Base.getproperty(mp::MPValue, name::Symbol)
    getproperty(getfield(mp, :prop), name)
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

@inline function difftype(mp::MPValue)
    hasproperty(mp, :∇∇N) && return hessian
    hasproperty(mp, :∇N)  && return gradient
    hasproperty(mp, :N)   && return identity
    error("unreachable")
end
@inline @propagate_inbounds value(::typeof(identity), f, x, args...) = (value(f, x, args...),)
@inline @propagate_inbounds value(::typeof(gradient), f, x, args...) = reverse(gradient(x -> (@_inline_meta; @_propagate_inbounds_meta; value(f, x, args...)), x, :all))
@inline @propagate_inbounds value(::typeof(hessian), f, x, args...) = reverse(hessian(x -> (@_inline_meta; @_propagate_inbounds_meta; value(f, x, args...)), x, :all))

@inline @propagate_inbounds set_shape_values!(mp::MPValue, ip, (N,)::Tuple{Any}) = (mp.N[ip]=N;)
@inline @propagate_inbounds set_shape_values!(mp::MPValue, ip, (N,∇N)::Tuple{Any,Any}) = (mp.N[ip]=N; mp.∇N[ip]=∇N;)
@inline @propagate_inbounds set_shape_values!(mp::MPValue, ip, (N,∇N,∇∇N)::Tuple{Any,Any,Any}) = (mp.N[ip]=N; mp.∇N[ip]=∇N; mp.∇∇N[ip]=∇∇N;)
@inline set_shape_values!(mp::MPValue, (N,)::Tuple{Any}) = copyto!(mp.N, N)
@inline set_shape_values!(mp::MPValue, (N,∇N)::Tuple{Any,Any}) = (copyto!(mp.N,N); copyto!(mp.∇N,∇N);)
@inline set_shape_values!(mp::MPValue, (N,∇N,∇∇N)::Tuple{Any,Any,Any}) = (copyto!(mp.N,N); copyto!(mp.∇N,∇N); copyto!(mp.∇∇N,∇∇N);)

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

function generate_mpvalues(::Type{Vec{dim, T}}, it::Interpolation, diff::Val, n::Int) where {dim, T}
    prop = map(create_property(Vec{dim, T}, it, diff)) do prop
        fill(zero(eltype(prop)), size(prop)..., n)
    end
    indices = fill(EmptyCartesianIndices(Val(dim)), n)
    It = typeof(it)
    Prop = typeof(prop)
    Indices = typeof(indices)
    ElType = Base._return_type(_getindex, Tuple{It, Prop, Indices, Int})
    MPValueVector{It, Prop, Indices, ElType}(it, prop, indices)
end
generate_mpvalues(::Type{Vec{dim}}, it::Interpolation, diff::Val, n::Int) where {dim} = generate_mpvalues(Vec{dim, Float64}, it, diff, n)

generate_mpvalues(::Type{V}, it::Interpolation, n::Int) where {V} = generate_mpvalues(V, it, Val(1), n)

Base.IndexStyle(::Type{<: MPValueVector}) = IndexLinear()
Base.size(x::MPValueVector) = size(getfield(x, :indices))

Base.propertynames(x::MPValueVector) = propertynames(getfield(x, :prop))
@inline function Base.getproperty(x::MPValueVector, name::Symbol)
    getproperty(getfield(x, :prop), name)
end

@inline interpolation(x::MPValueVector) = getfield(x, :it)

@inline function Base.getindex(x::MPValueVector, i::Integer)
    @boundscheck checkbounds(x, i)
    @inbounds _getindex(getfield(x, :it), getfield(x, :prop), getfield(x, :indices), i)
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
