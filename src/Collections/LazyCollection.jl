"""
    LazyCollections.LazyOperationType(f)

This needs to be overrided for custom operator.
Return `LazyAddLikeOperator()` or `LazyMulLikeOperator()`.
"""
abstract type LazyOperationType end
struct LazyAddLikeOperator <: LazyOperationType end
struct LazyMulLikeOperator <: LazyOperationType end
LazyOperationType(::Any) = LazyAddLikeOperator()
@pure function LazyOperationType(f::Function)
    Base.operator_precedence(Symbol(f)) ≥ Base.operator_precedence(:*) ?
        LazyMulLikeOperator() : LazyAddLikeOperator()
end

# add `Ref`s
lazyable(::LazyOperationType, c, ::Val) = Ref(c)
lazyable(::LazyOperationType, c::Base.RefValue, ::Val) = c
lazyable(::LazyAddLikeOperator, c::AbstractCollection{rank}, ::Val{rank}) where {rank} = c
lazyable(::LazyAddLikeOperator, c::AbstractCollection, ::Val) = throw(ArgumentError("addition like operation with different collections is not allowded"))
lazyable(::LazyMulLikeOperator, c::AbstractCollection{rank}, ::Val{rank}) where {rank} = c
lazyable(::LazyMulLikeOperator, c::AbstractCollection{0}, ::Val{1}) = Collection{1}(c) # 0 becomes 1 with other 1
@generated function lazyables(f, args...)
    rank = maximum(whichrank, args)
    if minimum(whichrank, args) == -1
        if rank != -1
            return :(throw(ArgumentError("rank=-1 collection cannot be computed with other rank collections.")))
        end
    end
    Expr(:tuple, [:(lazyable(LazyOperationType(f), args[$i], Val($rank))) for i in 1:length(args)]...)
end
lazyables(f, args′::Union{Base.RefValue, AbstractCollection{rank}}...) where {rank} = args′ # already "lazyabled"

# extract arguments without `Ref`
_extract_norefs(ret::Tuple) = ret
_extract_norefs(ret::Tuple, x::Ref, y...) = _extract_norefs(ret, y...)
_extract_norefs(ret::Tuple, x, y...) = _extract_norefs((ret..., x), y...)
extract_norefs(x...) = _extract_norefs((), x...)
extract_norefs(x::AbstractCollection...) = x

"""
    return_rank(f, args...)

Get returned rank.
"""
function return_rank(f, args...)
    args′ = extract_norefs(lazyables(f, args...)...)
    return_rank(LazyOperationType(f), args′...)
end
return_rank(::LazyAddLikeOperator, ::AbstractCollection{rank}...) where {rank} = rank
return_rank(::LazyMulLikeOperator, ::AbstractCollection{rank}...) where {rank} = rank
return_rank(::LazyMulLikeOperator, ::AbstractCollection{0}) = 0
return_rank(::LazyMulLikeOperator, ::AbstractCollection{0}, ::AbstractCollection{0}) = -1
return_rank(::LazyMulLikeOperator, ::AbstractCollection{0}, ::AbstractCollection{0}, x::AbstractCollection{0}...) =
    throw(ArgumentError("rank=-1 collections are used $(2+length(x)) times in multiplication"))
return_rank(::LazyMulLikeOperator, ::AbstractCollection{-1}) = -1
return_rank(::LazyMulLikeOperator, ::AbstractCollection{-1}, x::AbstractCollection{-1}...) =
    throw(ArgumentError("rank=-1 collections are used $(1+length(x)) times in multiplication"))

"""
    return_dims(f, args...)

Get returned dimensions.
"""
function return_dims(f, args...)
    args′ = extract_norefs(lazyables(f, args...)...)
    return_dims(LazyOperationType(f), args′...)
end
check_dims(x::Dims) = x
check_dims(x::Dims, y::Dims, z::Dims...) = (@assert x == y; check_dims(y, z...))
return_dims(::LazyAddLikeOperator, args::AbstractCollection{rank}...) where {rank} = check_dims(map(size, args)...)
return_dims(::LazyMulLikeOperator, args::AbstractCollection{rank}...) where {rank} = check_dims(map(size, args)...)
return_dims(::LazyMulLikeOperator, x::AbstractCollection{0}, y::AbstractCollection{0}) = (length(x), length(y))
return_dims(::LazyMulLikeOperator, x::AbstractCollection{-1}) = size(x)


struct LazyCollection{rank, F, Args <: Tuple, N} <: AbstractCollection{rank}
    f::F
    args::Args
    dims::NTuple{N, Int}
    function LazyCollection{rank, F, Args, N}(f::F, args::Args, dims::NTuple{N, Int}) where {rank, F, Args, N}
        new{rank::Int, F, Args, N}(f, args, dims)
    end
end

@inline function LazyCollection{rank}(f::F, args::Args, dims::NTuple{N, Int}) where {rank, F, Args, N}
    LazyCollection{rank, F, Args, N}(f, args, dims)
end

@generated function LazyCollection(f, args...)
    quote
        args′ = lazyables(f, args...)
        norefs = extract_norefs(args′...)
        rank = return_rank(f, norefs...)
        dims = return_dims(f, norefs...)
        LazyCollection{rank}(f, args′, dims)
    end
end
lazy(f, args...) = LazyCollection(f, args...)

Base.length(c::LazyCollection) = prod(c.dims)
Base.size(c::LazyCollection) = c.dims
Base.ndims(c::LazyCollection) = length(size(c))

@inline _getindex(c::AbstractCollection, i::Int) = (@_propagate_inbounds_meta; c[i])
@inline _getindex(c::Base.RefValue, i::Int) = c[]
@generated function Base.getindex(c::LazyCollection{<: Any, <: Any, Args, 1}, i::Int) where {Args}
    exps = [:(_getindex(c.args[$j], i)) for j in 1:length(Args.parameters)]
    quote
        @_inline_meta
        @boundscheck checkbounds(c, i)
        @inbounds c.f($(exps...))
    end
end

@generated function Base.getindex(c::LazyCollection{-1, <: Any, Args, 2}, ij::Vararg{Int, 2}) where {Args}
    count = 0
    exps = map(enumerate(Args.parameters)) do (k, T)
        T <: Base.RefValue && return :(c.args[$k][])
        T <: AbstractCollection{-1} && return :(c.args[$k][ij...])
        T <: AbstractCollection && return :(c.args[$k][ij[$(count += 1)]])
        error()
    end
    @assert count == 0 || count == 2
    quote
        @_inline_meta
        @_propagate_inbounds_meta
        @inbounds c.f($(exps...))
    end
end
@inline function Base.getindex(c::LazyCollection{-1, <: Any, <: Any, 2}, i::Int)
    @boundscheck checkbounds(c, i)
    @inbounds begin
        I = CartesianIndices(size(c))[i...]
        c[Tuple(I)...]
    end
end

# convert to array
# this is needed for matrix type because `collect` is called by default
function Base.Array(c::LazyCollection)
    v = first(c)
    A = Array{typeof(v)}(undef, size(c))
    for i in eachindex(A)
        @inbounds A[i] = c[i]
    end
    A
end

show_type_name(c::LazyCollection) = "LazyCollection{$(whichrank(c))}"


macro define_lazy_unary_operation(op)
    quote
        @inline $op(x::AbstractCollection) = lazy($op, x)
    end |> esc
end

macro define_lazy_binary_operation(op)
    quote
        @inline $op(c::AbstractCollection, x) = lazy($op, c, x)
        @inline $op(x, c::AbstractCollection) = lazy($op, x, c)
        @inline $op(x::AbstractCollection, y::AbstractCollection) = lazy($op, x, y)
    end |> esc
end

const unary_operations = [
    :(TensorValues.∇),
    :(TensorValues.symmetric),
    :(TensorValues.tr),
    :(TensorValues.vol),
    :(TensorValues.mean),
    :(TensorValues.det),
    :(TensorValues.Tensor2D),
    :(TensorValues.Tensor3D),
    :(MaterialModels.volumetric_stress),
    :(MaterialModels.deviatoric_stress),
    :(MaterialModels.volumetric_strain),
    :(MaterialModels.deviatoric_strain),
    :(MaterialModels.infinitesimal_strain),
    :(Base.:+),
    :(Base.:-),
    :(Base.:log),
    :(Base.:log10),
    :(LinearAlgebra.norm),
]

const binary_operations = [
    :(Base.:+),
    :(Base.:-),
    :(Base.:*),
    :(Base.:/),
    :(Base.:^),
    :(TensorValues.:⋅),
    :(TensorValues.:⊗),
    :(TensorValues.:×),
    :(TensorValues.:⊡),
    :(TensorValues.valgrad),
    :(TensorValues._otimes_),
]

for op in unary_operations
    @eval @define_lazy_unary_operation $op
end

for op in binary_operations
    @eval @define_lazy_binary_operation $op
end

LazyOperationType(::typeof(TensorValues.dotdot)) = LazyMulLikeOperator()
