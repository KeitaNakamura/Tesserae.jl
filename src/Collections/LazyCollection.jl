"""
    LazyCollection
"""
struct LazyCollection{rank, F, Args <: Tuple, N} <: AbstractCollection{rank}
    f::F
    args::Args
    dims::NTuple{N, Int}
end

getrank(::Type{<: AbstractCollection{rank}}) where {rank} = rank
getrank(::Type) = -100 # just use low value
maxrank(args...) = maximum(getrank, args)

addref(c::AbstractCollection{rank}, ::Val{rank}) where {rank} = c
addref(c, ::Val{rank}) where {rank} = Ref(c)

extract_norefs(ret::Tuple) = ret
extract_norefs(ret::Tuple, x::Ref, y...) = extract_norefs(ret, y...)
extract_norefs(ret::Tuple, x, y...) = extract_norefs((ret..., x), y...)

check_length(x::Int) = (x,)
check_length(x::Int, y::Int, z::Int...) = (@assert x == y; check_length(y, z...))
combine_dims(xs::AbstractCollection{rank}...) where {rank} = check_length(map(length, xs)...)
combine_dims(x::AbstractCollection{0}...) = map(length, x)

function LazyCollection{rank}(f::F, args::Args, dims::NTuple{N, Int}) where {rank, F, Args <: Tuple, N}
    LazyCollection{rank, F, Args, N}(f, args, dims)
end
@generated function LazyCollection(f::F, args::Args) where {F, Args <: Tuple}
    rank = maxrank(Args.parameters...)
    quote
        newargs = broadcast(addref, args, Val($rank))
        norefs = extract_norefs((), newargs...)
        dims = combine_dims(norefs...)
        LazyCollection{$rank}(f, newargs, dims)
    end
end
lazy(f, args...) = LazyCollection(f, args)

Base.length(c::LazyCollection) = prod(c.dims)
Base.size(c::LazyCollection) = c.dims
Base.ndims(c::LazyCollection) = length(size(c))

@inline _getindex(ret::Tuple, x::Tuple{}, I::Int...) = ret
@inline function _getindex(ret::Tuple, x::Tuple{Base.RefValue, Vararg}, I::Int...)
    @_propagate_inbounds_meta
    v = x[1][]
    _getindex((ret..., v), Base.tail(x), I...)
end
@inline function _getindex(ret::Tuple, x::Tuple{AbstractCollection, Vararg}, I::Int)
    @_propagate_inbounds_meta
    v = x[1][I]
    _getindex((ret..., v), Base.tail(x), I)
end
@inline function _getindex(ret::Tuple, x::Tuple{AbstractCollection{0}, Vararg}, I::Int...)
    @_propagate_inbounds_meta
    v = x[1][I[1]]
    _getindex((ret..., v), Base.tail(x), Base.tail(I)...)
end

@inline function Base.getindex(c::LazyCollection{<: Any, <: Any, <: Any, 1}, i::Int)
    @boundscheck checkbounds(c, i)
    @inbounds begin
        args = _getindex((), c.args, i)
        c.f(args...)
    end
end
# cartesian indexing for matrix form
@inline function Base.getindex(c::LazyCollection{<: Any, <: Any, <: Any, N}, I::Vararg{Int, N}) where {N}
    @_propagate_inbounds_meta
    args = _getindex((), c.args, I...)
    c.f(args...)
end
@inline Base.getindex(c::LazyCollection{<: Any, <: Any, <: Any, N}, I::CartesianIndex{N}) where {N} = getindex(c, Tuple(I)...)
@inline Base.getindex(c::LazyCollection{<: Any, <: Any, <: Any, 1}, I::CartesianIndex{1}) where {N} = getindex(c, Tuple(I)...)
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
function Base.show(io::IO, c::LazyCollection{-1})
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

macro define_unary_operation(op)
    quote
        @inline $op(c::AbstractCollection) = lazy($op, c)
    end |> esc
end

macro define_binary_operation(op)
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
    @eval @define_unary_operation $op
end

for op in binary_operations
    @eval @define_binary_operation $op
end
