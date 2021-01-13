"""
    AbstractCollection{rank, T}

Supertype for collections.
"""
abstract type AbstractCollection{rank} end

Base.eltype(c::AbstractCollection) = isempty(c) ? Union{} : typeof(first(c)) # try to getindex

function Base.fill!(c::AbstractCollection, v)
    for i in eachindex(c)
        @inbounds c[i] = v
    end
end

# checkbounds
@inline Base.checkbounds(::Type{Bool}, c::AbstractCollection, i) =
    checkindex(Bool, eachindex(c), i)
@inline Base.checkbounds(::Type{Bool}, c::AbstractCollection, i::CartesianIndex) =
    Base.checkbounds_indices(Bool, axes(c), (i,))
@inline Base.checkbounds(c::AbstractCollection, i) =
    checkbounds(Bool, c, i) ? nothing : throw(BoundsError(c, i))

# getindex
Base.IndexStyle(::Type{<: AbstractCollection}) = IndexLinear()
Base.size(c::AbstractCollection) = (length(c),)
Base.eachindex(c::AbstractCollection) = Base.OneTo(lastindex(c))
Base.firstindex(c::AbstractCollection) = 1
Base.lastindex(c::AbstractCollection) = length(c)
@inline Base.getindex(c::AbstractCollection, i::CartesianIndex{1}) = (@_propagate_inbounds_meta; c[i[1]])

# iterate
@inline Base.iterate(c::AbstractCollection, i = 1) = (i % UInt) - 1 < length(c) ? (@inbounds c[i], i + 1) : nothing

# convert to Array
Base.Array(c::AbstractCollection) = collect(c)

function Base.isassigned(c::AbstractCollection, i)
    try
        c[i]
        true
    catch e
        if isa(e, BoundsError) || isa(e, UndefRefError)
            return false
        else
            rethrow()
        end
    end
end

# broadcast
Broadcast.broadcastable(c::AbstractCollection) = c
Broadcast.BroadcastStyle(::Type{<: AbstractCollection}) = Broadcast.DefaultArrayStyle{1}()

function Base.show(io::IO, c::AbstractCollection{rank}) where {rank}
    io = IOContext(io, :typeinfo => eltype(c))
    print(io, "<", length(c), " × ", eltype(c), ">[")
    join(io, [isassigned(c, i) ? sprint(show, c[i]; context=io) : "#undef" for i in eachindex(c)], ", ")
    print(io, "]")
    if !get(io, :compact, false)
        print(io, " with rank=$rank")
    end
end


"""
    Collection(x, [Val(rank)])
"""
struct Collection{rank, V <: Union{AbstractVector, AbstractCollection}} <: AbstractCollection{rank}
    parent::V
end

# constructors
Collection{rank}(v::V) where {rank, V} = Collection{rank, V}(v)
Collection(v) = Collection{1}(v)
"""
    collection(x, [Val(rank)])

Create collection with `x`.
"""
collection(v, ::Val{rank} = Val(1)) where {rank} = Collection{rank}(v)

Base.parent(c::Collection) = c.parent

# needs to be implemented for AbstractCollection
Base.length(c::Collection) = length(parent(c))
@inline function Base.getindex(c::Collection, i::Integer)
    @boundscheck checkbounds(c, i)
    @inbounds parent(c)[i]
end
@inline function Base.setindex!(c::Collection, v, i::Integer)
    @boundscheck checkbounds(c, i)
    @inbounds parent(c)[i] = v
    c
end


"""
    LazyCollection
"""
struct LazyCollection{rank, Bc} <: AbstractCollection{rank}
    bc::Bc
end

# constructors
LazyCollection{rank}(bc::Bc) where {rank, Bc} =
    LazyCollection{rank, Bc}(bc)

Base.length(c::LazyCollection) = length(c.bc)
Base.size(c::LazyCollection) = size(c.bc) # needs to be override
Base.ndims(c::LazyCollection) = length(axes(c.bc))
Base.getindex(c::LazyCollection, i::Int) = (@_propagate_inbounds_meta; c.bc[i])
function Base.first(c::LazyCollection)
    ndims(c) == 1 ? c.bc[1] : c.bc[1,1]
end

Broadcast.broadcastable(c::LazyCollection) = c.bc

# convert to array
# this is needed for matrix type because `collect` is called by default
Base.Array(c::LazyCollection) = copy(c.bc)

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


struct AdjointCollection{rank, P <: AbstractCollection{rank}} <: AbstractCollection{rank}
    parent::P
end
Base.parent(c::AdjointCollection) = c.parent
LinearAlgebra.adjoint(c::AbstractCollection) = AdjointCollection(c)

# getindex
Base.IndexStyle(::Type{<: AdjointCollection}) = IndexLinear()
Base.length(c::AdjointCollection) = length(parent(c))
Base.size(c::AdjointCollection) = (1, length(c))
@inline Base.getindex(c::AdjointCollection, i::Int) = (@_propagate_inbounds_meta; parent(c)[i])
@inline Base.getindex(c::AdjointCollection, i::CartesianIndex{2}) = (@_propagate_inbounds_meta; @assert i[1] == 1; parent(c)[i[2]])

# Broadcast
Broadcast.broadcastable(c::AdjointCollection) = c
Broadcast.BroadcastStyle(::Type{<: AdjointCollection}) = Broadcast.DefaultArrayStyle{2}()

##############
# Operations #
##############
# rank= 1: nodal value
# rank= 2: point value
# rank= 0: nodal value, but used only for shape function
# rank=-1: computation with scalar type is allowed

# unary
function lazy(op, c::AbstractCollection{rank}) where {rank}
    LazyCollection{rank}(broadcasted(op, c))
end

# binary
lazy(op, c::AbstractCollection{rank}, x) where {rank} = LazyCollection{rank}(broadcasted(op, c, Ref(x)))
lazy(op, x, c::AbstractCollection{rank}) where {rank} = LazyCollection{rank}(broadcasted(op, Ref(x), c))
lazy(op, x::AbstractCollection{rank}, y::AbstractCollection{rank}) where {rank} =
    LazyCollection{rank}(broadcasted(op, x, y))
function lazy(op, x::AbstractCollection{L1}, y::AbstractCollection{L2}) where {L1, L2}
    L1 > L2 ? LazyCollection{L1}(broadcasted(op, x, Ref(y))) :
              LazyCollection{L2}(broadcasted(op, Ref(x), y))
end
# special cases and errors
operation_error(op, rank1, rank2) = throw(ArgumentError("wrong collection operation with $op(rank=$rank1, rank=$rank2)"))
## Nᵢ[p] * vᵢ[p]
lazy(op, x::AbstractCollection{0}, y::AbstractCollection{1}) = LazyCollection{-1}(broadcasted(op, x, y))
lazy(op, x::AbstractCollection{1}, y::AbstractCollection{0}) = LazyCollection{-1}(broadcasted(op, x, y))
## Nᵢ[p] * Nᵢ[p]
lazy(op::typeof(*), x::AbstractCollection{0}, y::AbstractCollection{0}) = LazyCollection{-1}(broadcasted(op, x, y'))
lazy(op::typeof(⊗), x::AbstractCollection{0}, y::AbstractCollection{0}) = LazyCollection{-1}(broadcasted(op, x, y'))
## errors (with point value)
lazy(op, x::AbstractCollection{2}, y::AbstractCollection{1}) = operation_error(op, 2, 1)
lazy(op, x::AbstractCollection{2}, y::AbstractCollection{0}) = operation_error(op, 2, 0)
lazy(op, x::AbstractCollection{1}, y::AbstractCollection{2}) = operation_error(op, 1, 2)
lazy(op, x::AbstractCollection{0}, y::AbstractCollection{2}) = operation_error(op, 0, 2)
## errors (with rank=-1 collection)
lazy(op, x::AbstractCollection{-1}, y::AbstractCollection{0}) = operation_error(op, -1, 0)
lazy(op, x::AbstractCollection{-1}, y::AbstractCollection{1}) = operation_error(op, -1, 1)
lazy(op, x::AbstractCollection{-1}, y::AbstractCollection{2}) = operation_error(op, -1, 2)
lazy(op, x::AbstractCollection{0}, y::AbstractCollection{-1}) = operation_error(op, 0, -1)
lazy(op, x::AbstractCollection{1}, y::AbstractCollection{-1}) = operation_error(op, 1, -1)
lazy(op, x::AbstractCollection{2}, y::AbstractCollection{-1}) = operation_error(op, 2, -1)

# ternary
TensorValues.dotdot(u::AbstractCollection, x, v::AbstractCollection) = lazy(dotdot, u, x, v)
lazy(::typeof(TensorValues.dotdot), u::AbstractCollection{0}, x::SymmetricTensor{4}, v::AbstractCollection{0}) = LazyCollection{-1}(broadcasted(dotdot, u, Ref(x), v'))
lazy(::typeof(TensorValues.dotdot), u::AbstractCollection{2}, x::SymmetricTensor{4}, v::AbstractCollection{2}) = LazyCollection{2}(broadcasted(dotdot, u, Ref(x), v))
lazy(::typeof(TensorValues.dotdot), u::AbstractCollection{2}, x::AbstractCollection{2}, v::AbstractCollection{2}) = LazyCollection{2}(broadcasted(dotdot, u, x, v))

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
    :(TensorValues.divergence),
    :(TensorValues.tr),
    :(TensorValues.vol),
    :(TensorValues.mean),
    :(TensorValues.det),
    :(TensorValues.tensor2x2),
    :(TensorValues.tensor3x3),
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


struct CollectionView{rank, C <: AbstractCollection{rank}, I} <: AbstractCollection{rank}
    parent::C
    indices::I
end

@inline function Base.view(c::AbstractCollection, I::Vararg{Any, N}) where {N}
    @boundscheck checkbounds(c, I...)
    CollectionView(c, I...)
end

Base.parent(c::AbstractCollection) = c.parent
Base.parentindices(c::AbstractCollection) = c.indices

Base.length(c::CollectionView) = length(parentindices(c))
Base.getindex(c::CollectionView, i::Int) = (@_propagate_inbounds_meta; parent(c)[parentindices(c)[i]])
