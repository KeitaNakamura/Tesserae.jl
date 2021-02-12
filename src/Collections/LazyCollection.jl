"""
    LazyCollection
"""
struct LazyCollection{rank, mulprec, F, Args <: Tuple, N} <: AbstractCollection{rank}
    f::F
    args::Args
    dims::NTuple{N, Int}
    function LazyCollection{rank, mulprec, F, Args, N}(f::F, args::Args, dims::NTuple{N, Int}) where {rank, mulprec, F, Args, N}
        new{rank::Int, mulprec::Bool, F, Args, N}(f, args, dims)
    end
end

@inline function LazyCollection{rank, mulprec}(f::F, args::Args, dims::NTuple{N, Int}) where {rank, mulprec, F, Args, N}
    LazyCollection{rank, mulprec, F, Args, N}(f, args, dims)
end

getrank(::Type{<: AbstractCollection{rank}}) where {rank} = rank
getrank(::Type) = -100 # just use low value

# extract arguments without `Ref`
_extract_norefs(ret::Tuple) = ret
_extract_norefs(ret::Tuple, x::Ref, y...) = _extract_norefs(ret, y...)
_extract_norefs(ret::Tuple, x, y...) = _extract_norefs((ret..., x), y...)
extract_norefs(x...) = _extract_norefs((), x...)

# check if all `length`s are the same
check_length(x::Int) = x
check_length(x::Int, y::Int, z...) = (@assert x == y; check_length(y, z...))
# guess common dims
@generated function combine_dims(::Val{mulprec}, args′...) where {mulprec}
    quote
        @_inline_meta
        norefs = extract_norefs(args′...)
        $(mulprec ? :(map(length, norefs)) : :(tuple(check_length(map(length, norefs)...))))
    end
end

# add `Ref`s
lazyable(c::AbstractCollection{rank}, ::Val{rank}) where {rank} = c
lazyable(c::AbstractCollection{0}, ::Val{1}) = Collection{1}(c) # this is special case for `N * vᵢ`
lazyable(c, ::Val) = Ref(c)

# Constructor controlling `mulprec`
@generated function LazyCollection(::Val{mulprec}, f, args...) where {mulprec}
    rank = maximum(getrank, args)
    quote
        @_inline_meta
        args′ = broadcast(lazyable, args, Val($rank))
        dims = combine_dims(Val(mulprec), args′...)
        LazyCollection{$rank, mulprec}(f, args′, dims)
    end
end

# Constructor (`mulprec` is guess from `f`)
multiply_precedence(x::Symbol) = Base.operator_precedence(x) ≥ Base.operator_precedence(:*)
@generated function LazyCollection(f, args...)
    mulprec = multiply_precedence(Symbol(f.instance))
    rank = maximum(getrank, args)
    if rank != 0
        mulprec = false
    end
    quote
        @_inline_meta
        LazyCollection(Val($mulprec), f, args...)
    end
end
lazy(f, args...) = LazyCollection(f, args...)

Base.length(c::LazyCollection) = prod(c.dims)
Base.size(c::LazyCollection) = c.dims
Base.ndims(c::LazyCollection) = length(size(c))

@generated function Base.getindex(c::LazyCollection{<: Any, mulprec, <: Any, Args, N}, I::Vararg{Int, N}) where {mulprec, Args, N}
    count = 0
    exps = map(enumerate(Args.parameters)) do (i, T)
        T <: Base.RefValue && return :(c.args[$i][])
        if T <: AbstractCollection{0} && mulprec
            return :(c.args[$i][I[$(count += 1)]])
        end
        if T <: AbstractCollection
            @assert N == 1
            return :(c.args[$i][first(I)])
        end
        error()
    end
    quote
        @_inline_meta
        $(N == 1 ? :(@boundscheck checkbounds(c, I...)) : :(@_propagate_inbounds_meta))
        @inbounds c.f($(exps...))
    end
end
@inline Base.getindex(c::LazyCollection{<: Any, <: Any, <: Any, <: Any, N}, I::CartesianIndex{N}) where {N} = getindex(c, Tuple(I)...)
@inline Base.getindex(c::LazyCollection{<: Any, <: Any, <: Any, <: Any, 1}, I::CartesianIndex{1}) = getindex(c, Tuple(I)...) # to fix ambiguity
# linear indexing for matrix form
@inline function Base.getindex(c::LazyCollection, i::Int)
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
    for I in CartesianIndices(A)
        @inbounds A[I] = c[I]
    end
    A
end

# show
_typetostring(::Type{<: LazyCollection{rank}}) where {rank} = "LazyCollection{$rank}"
_typetostring(T::Type) = "$T"
_typetostring(::Type{Union{}}) = "Any"
function Base.show(io::IO, c::LazyCollection{rank}) where {rank}
    io = IOContext(io, :typeinfo => eltype(c))
    print(io, "<", length(c), " × ", _typetostring(eltype(c)), ">[")
    join(io, [sprint(show, c[i]; context = IOContext(io, :compact => true)) for i in eachindex(c)], ", ")
    print(io, "]")
    if !get(io, :compact, false)
        print(io, " with rank=$rank")
    end
end
function Base.show(io::IO, c::LazyCollection{0})
    join(io, size(c), "×")
    print(io, " Array(collection) = ", Array(c))
end


##############
# Operations #
##############
# rank= 1: nodal value
# rank= 2: point value
# rank= 0: nodal value, but used only for shape function
# rank=-1: computation with scalar type is allowed

#=
# unary
function lazy(op, c::AbstractCollection{rank}) where {rank}
    LazyCollection{rank}(op, c)
end

# binary
lazy(op, c::AbstractCollection{rank}, x) where {rank} = LazyCollection{rank}(op, c, x)
lazy(op, x, c::AbstractCollection{rank}) where {rank} = LazyCollection{rank}(op, x, c)
lazy(op, x::AbstractCollection{rank}, y::AbstractCollection{rank}) where {rank} =
    LazyCollection{rank}(op, x, y)
function lazy(op, x::AbstractCollection{L1}, y::AbstractCollection{L2}) where {L1, L2}
    L1 > L2 ? LazyCollection{L1}(op, x, y) :
              LazyCollection{L2}(op, x, y)
end
# special cases and errors
operation_error(op, rank1, rank2) = throw(ArgumentError("wrong collection operation with $op(rank=$rank1, rank=$rank2)"))
## Nᵢ[p] * vᵢ[p]
lazy(op, x::AbstractCollection{0}, y::AbstractCollection{1}) = LazyCollection{-1}(op, x, y)
lazy(op, x::AbstractCollection{1}, y::AbstractCollection{0}) = LazyCollection{-1}(op, x, y)
## Nᵢ[p] * Nᵢ[p]
lazy(op::typeof(*), x::AbstractCollection{0}, y::AbstractCollection{0}) = LazyCollection{-1}(op, x, y)
lazy(op::typeof(⊗), x::AbstractCollection{0}, y::AbstractCollection{0}) = LazyCollection{-1}(op, x, y)
lazy(op::typeof(⋅), x::AbstractCollection{0}, y::AbstractCollection{0}) = LazyCollection{-1}(op, x, y)
lazy(op::typeof(+), x::AbstractCollection{0}, y::AbstractCollection{0}) = LazyCollection{0}(op, x, y)
lazy(op::typeof(-), x::AbstractCollection{0}, y::AbstractCollection{0}) = LazyCollection{0}(op, x, y)
lazy(op, x::AbstractCollection{0}, y::AbstractCollection{0}) = operation_error(op, 0, 0)
## errors (point value can be calculated with only point value)
lazy(op, x::AbstractCollection{2}, y::AbstractCollection{1}) = operation_error(op, 2, 1)
lazy(op, x::AbstractCollection{2}, y::AbstractCollection{0}) = operation_error(op, 2, 0)
lazy(op, x::AbstractCollection{1}, y::AbstractCollection{2}) = operation_error(op, 1, 2)
lazy(op, x::AbstractCollection{0}, y::AbstractCollection{2}) = operation_error(op, 0, 2)
## errors (rank=-1 collection cannot be calculated with any collections)
lazy(op, x::AbstractCollection{-1}, y::AbstractCollection{0}) = operation_error(op, -1, 0)
lazy(op, x::AbstractCollection{-1}, y::AbstractCollection{1}) = operation_error(op, -1, 1)
lazy(op, x::AbstractCollection{-1}, y::AbstractCollection{2}) = operation_error(op, -1, 2)
lazy(op, x::AbstractCollection{0}, y::AbstractCollection{-1}) = operation_error(op, 0, -1)
lazy(op, x::AbstractCollection{1}, y::AbstractCollection{-1}) = operation_error(op, 1, -1)
lazy(op, x::AbstractCollection{2}, y::AbstractCollection{-1}) = operation_error(op, 2, -1)

# ternary
TensorValues.dotdot(u::AbstractCollection, x, v::AbstractCollection) = lazy(dotdot, u, x, v)
lazy(::typeof(TensorValues.dotdot), u::AbstractCollection{0}, x::SymmetricFourthOrderTensor, v::AbstractCollection{0}) = LazyCollection{-1}(dotdot, u, x, v')
lazy(::typeof(TensorValues.dotdot), u::AbstractCollection{2}, x::SymmetricFourthOrderTensor, v::AbstractCollection{2}) = LazyCollection{2}(dotdot, u, x, v)
lazy(::typeof(TensorValues.dotdot), u::AbstractCollection{2}, x::AbstractCollection{2}, v::AbstractCollection{2}) = LazyCollection{2}(dotdot, u, x, v)
=#

macro define_lazy_operation(op)
    quote
        @inline $op(x::AbstractCollection, y::AbstractCollection...) = lazy($op, x, y...)
        @inline $op(c::AbstractCollection, x) = lazy($op, c, x)
        @inline $op(x, c::AbstractCollection) = lazy($op, x, c)
    end |> esc
end

const operations = [
    :(TensorValues.:⋅),
    :(TensorValues.:⊗),
    :(TensorValues.:×),
    :(TensorValues.:⊡),
    :(TensorValues.valgrad),
    :(TensorValues._otimes_),
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
    :(Base.:*),
    :(Base.:/),
    :(Base.:^),
    :(Base.:log),
    :(Base.:log10),
    :(LinearAlgebra.norm),
]

for op in operations
    @eval @define_lazy_operation $op
end
