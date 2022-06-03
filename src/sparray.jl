struct SpPattern{dim} <: AbstractArray{Bool, dim}
    indices::Array{Int, dim}
end

SpPattern(dims::Tuple{Vararg{Int}}) = SpPattern(fill(-1, dims))
SpPattern(dims::Int...) = SpPattern(dims)

Base.size(spat::SpPattern) = size(spat.indices)
Base.IndexStyle(::Type{<: SpPattern}) = IndexLinear()

@inline Base.getindex(spat::SpPattern, i::Int) = (@_propagate_inbounds_meta; spat.indices[i] !== -1)

function update_sppattern!(spat::SpPattern, mask::AbstractArray{Bool})
    @assert size(spat) == size(mask)
    count = 0
    @inbounds for i in eachindex(spat)
        spat.indices[i] = (mask[i] ? count += 1 : -1)
    end
    count
end


"""
    SpArray{T}(dims...)

`SpArray` is a kind of sparse array, but it is not allowed to freely change the value like `Array`.
For example, trying to `setindex!` doesn't change anything without any errors as

```jldoctest sparray
julia> A = Marble.SpArray{Float64}(5,5)
5×5 Marble.SpArray{Float64, 2, Vector{Float64}}:
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅

julia> A[1,1]
0.0

julia> A[1,1] = 2
2

julia> A[1,1]
0.0
```

This is because the index `(1,1)` is not activated yet.
To activate the index, update sparsity pattern by `update_sppattern!(A, spat)`.

```jl sparray
julia> spat = falses(5,5); spat[1,1] = true; spat
5×5 BitMatrix:
 1  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0

julia> update_sppattern!(A, spat)
5×5 Marble.SpArray{Float64, 2, Vector{Float64}}:
 2.17321e-314  ⋅  ⋅  ⋅  ⋅
  ⋅            ⋅  ⋅  ⋅  ⋅
  ⋅            ⋅  ⋅  ⋅  ⋅
  ⋅            ⋅  ⋅  ⋅  ⋅
  ⋅            ⋅  ⋅  ⋅  ⋅

julia> A[1,1] = 2; A[1,1]
2.0
```

Although the inactive indices return zero value when using `getindex`,
the behaviors in array calculation is similar to `missing` rather than zero:

```jldoctest sparray; setup = :(spat=falses(5,5); spat[1,1]=true; update_sppattern!(A, spat); A[1,1]=2)
julia> A
5×5 Marble.SpArray{Float64, 2, Vector{Float64}}:
 2.0  ⋅  ⋅  ⋅  ⋅
  ⋅   ⋅  ⋅  ⋅  ⋅
  ⋅   ⋅  ⋅  ⋅  ⋅
  ⋅   ⋅  ⋅  ⋅  ⋅
  ⋅   ⋅  ⋅  ⋅  ⋅

julia> C = rand(5,5)
5×5 Matrix{Float64}:
 0.579862   0.639562  0.566704  0.870539  0.526344
 0.411294   0.839622  0.536369  0.962715  0.0779683
 0.972136   0.967143  0.711389  0.15118   0.966197
 0.0149088  0.789764  0.103929  0.715355  0.666558
 0.520355   0.696041  0.806704  0.939548  0.131026

julia> A + C
5×5 Marble.SpArray{Float64, 2, Vector{Float64}}:
 2.57986  ⋅  ⋅  ⋅  ⋅
  ⋅       ⋅  ⋅  ⋅  ⋅
  ⋅       ⋅  ⋅  ⋅  ⋅
  ⋅       ⋅  ⋅  ⋅  ⋅
  ⋅       ⋅  ⋅  ⋅  ⋅

julia> 3A
5×5 Marble.SpArray{Float64, 2, Vector{Float64}}:
 6.0  ⋅  ⋅  ⋅  ⋅
  ⋅   ⋅  ⋅  ⋅  ⋅
  ⋅   ⋅  ⋅  ⋅  ⋅
  ⋅   ⋅  ⋅  ⋅  ⋅
  ⋅   ⋅  ⋅  ⋅  ⋅
```

Thus, inactive indices are propagated as

```jldoctest sparray
julia> B = Marble.SpArray{Float64}(5,5);

julia> fill!(spat, false); spat[3,3] = true; update_sppattern!(B, spat); B[3,3] = 8.0;

julia> B
5×5 Marble.SpArray{Float64, 2, Vector{Float64}}:
 ⋅  ⋅   ⋅   ⋅  ⋅
 ⋅  ⋅   ⋅   ⋅  ⋅
 ⋅  ⋅  8.0  ⋅  ⋅
 ⋅  ⋅   ⋅   ⋅  ⋅
 ⋅  ⋅   ⋅   ⋅  ⋅

julia> A + B
5×5 Marble.SpArray{Float64, 2, Vector{Float64}}:
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
```
"""
struct SpArray{T, dim, V <: AbstractVector{T}} <: AbstractArray{T, dim}
    data::V
    spat::SpPattern{dim}
    parent::Bool
    stamp::Base.RefValue{Float64} # only used when constructing `SpArray` by `generate_gridstate`
end

function SpArray{T}(dims::Tuple{Vararg{Int}}) where {T}
    data = Vector{T}(undef, prod(dims))
    spat = SpPattern(dims)
    SpArray(data, spat, true, Ref(NaN))
end
SpArray{T}(dims::Int...) where {T} = SpArray{T}(dims)

Base.IndexStyle(::Type{<: SpArray}) = IndexLinear()
Base.size(x::SpArray) = size(x.spat)

get_stamp(x::SpArray) = getfield(x, :stamp)[]
get_sppattern(x::SpArray) = getfield(x, :spat)

# handle `StructVector`
Base.propertynames(x::SpArray{<: Any, <: Any, <: StructVector}) = (:data, :spat, :stamp, :parent, propertynames(x.data)...)
function Base.getproperty(x::SpArray{<: Any, <: Any, <: StructVector}, name::Symbol)
    name == :data   && return getfield(x, :data)
    name == :spat   && return getfield(x, :spat)
    name == :stamp  && return getfield(x, :stamp)
    name == :parent && return getfield(x, :parent)
    SpArray(getproperty(getfield(x, :data), name), getfield(x, :spat), false, getfield(x, :stamp))
end

# return zero if the index is not active
@inline function Base.getindex(x::SpArray, i::Int)
    @boundscheck checkbounds(x, i)
    spat = x.spat
    index = spat.indices[i]
    @inbounds index !== -1 ? x.data[index] : zero_recursive(eltype(x))
end

# do nothing if the index is not active (don't throw error!!)
@inline function Base.setindex!(x::SpArray, v, i::Int)
    @boundscheck checkbounds(x, i)
    spat = x.spat
    @inbounds begin
        index = spat.indices[i]
        index === -1 && return x
        x.data[index] = v
    end
    x
end

# faster than using `setindex!(dest, dest + getindex(src, i))` when using `SpArray`
# since the index is checked only once
@inline function add!(x::SpArray, v, i)
    @boundscheck checkbounds(x, i)
    spat = x.spat
    @inbounds begin
        index = spat.indices[i]
        index === -1 && return x
        x.data[index] += v
    end
    x
end
@inline function add!(x::AbstractArray, v, i)
    @boundscheck checkbounds(x, i)
    @inbounds x[i] += v
    x
end

fillzero!(x::SpArray) = (fillzero!(x.data); x)

function update_sppattern!(x::SpArray, spat::AbstractArray{Bool})
    @assert x.parent
    @assert size(x) == size(spat)
    n = update_sppattern!(x.spat, spat)
    resize!(x.data, n)
    x.stamp[] = NaN
    x
end

#############
# Broadcast #
#############

Broadcast.BroadcastStyle(::Type{<: SpArray}) = ArrayStyle{SpArray}()

@generated function extract_sppatterns(args::Vararg{Any, N}) where {N}
    exps = []
    for i in 1:N
        if args[i] <: SpArray
            push!(exps, :(args[$i].spat))
        elseif (args[i] <: AbstractArray) && !(args[i] <: AbstractTensor)
            push!(exps, :nothing)
        end
    end
    quote
        tuple($(exps...))
    end
end
identical(x, ys...) = all(y -> y === x, ys)

_getspat(x::SpArray) = x.spat
_getspat(x::Any) = true
function Base.similar(bc::Broadcasted{ArrayStyle{SpArray}}, ::Type{ElType}) where {ElType}
    dims = size(bc)
    spat = BitArray(undef, dims)
    broadcast!(&, spat, map(_getspat, bc.args)...)
    A = SpArray{ElType}(dims)
    update_sppattern!(A, spat)
    A
end

_getdata(x::SpArray) = x.data
_getdata(x::Any) = x
function Base.copyto!(dest::SpArray, bc::Broadcasted{ArrayStyle{SpArray}})
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    bc′ = Broadcast.flatten(bc)
    if identical(extract_sppatterns(dest, bc′.args...)...)
        broadcast!(bc′.f, _getdata(dest), map(_getdata, bc′.args)...)
    else
        copyto!(dest, convert(Broadcasted{Nothing}, bc′))
    end
    dest
end

function Base.copyto!(dest::SpArray, bc::Broadcasted{ThreadedStyle})
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    bc′ = Broadcast.flatten(only(bc.args))
    if identical(extract_sppatterns(dest, bc′.args...)...)
        _copyto!(_getdata(dest), broadcasted(dot_threads, broadcasted(bc′.f, map(_getdata, bc′.args)...)))
    else
        _copyto!(dest, broadcasted(dot_threads, bc′))
    end
    dest
end

###############
# Custom show #
###############

struct CDot end
Base.show(io::IO, x::CDot) = print(io, "⋅")

struct ShowSpArray{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    parent::A
end
Base.size(x::ShowSpArray) = size(x.parent)
Base.axes(x::ShowSpArray) = axes(x.parent)
@inline function Base.getindex(x::ShowSpArray, i::Int...)
    @_propagate_inbounds_meta
    p = x.parent
    p.spat[i...] ? maybecustomshow(p[i...]) : CDot()
end
maybecustomshow(x) = x
maybecustomshow(x::SpArray) = ShowSpArray(x)

Base.summary(io::IO, x::ShowSpArray) = summary(io, x.parent)
Base.show(io::IO, mime::MIME"text/plain", x::SpArray) = show(io, mime, ShowSpArray(x))
Base.show(io::IO, x::SpArray) = show(io, ShowSpArray(x))
