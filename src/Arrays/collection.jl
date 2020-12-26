"""
    AbstractCollection{rank, T}

Supertype for collections.
"""
abstract type AbstractCollection{rank, T} end

Base.IndexStyle(::Type{<: AbstractCollection}) = IndexLinear()
Base.eltype(::AbstractCollection{rank, ElType}) where {rank, ElType} = ElType

Base.size(c::AbstractCollection) = (length(c),)
Base.eachindex(c::AbstractCollection) = Base.OneTo(length(c))

@inline function Base.checkbounds(::Type{Bool}, c::AbstractCollection, i::Integer)
    checkindex(Bool, Base.OneTo(length(c)), i)
end

@inline function Base.checkbounds(::Type{Bool}, c::AbstractCollection, i::CartesianIndex)
    Base.checkbounds_indices(Bool, axes(c), (i,))
end

@inline function Base.checkbounds(c::AbstractCollection, i)
    checkbounds(Bool, c, i) || throw(BoundsError(c, i))
    nothing
end

@inline Base.getindex(c::AbstractCollection, i::CartesianIndex{1}) = (@_propagate_inbounds_meta; c[i[1]])

@inline Base.iterate(c::AbstractCollection, i = 1) = (i % UInt) - 1 < length(c) ? (@inbounds c[i], i + 1) : nothing

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

# used for broadcast
function Broadcast.extrude(c::AbstractCollection)
    Broadcast.Extruded(c, Broadcast.newindexer(c)...)
end

Broadcast.broadcastable(c::AbstractCollection) = c
Broadcast.BroadcastStyle(::Type{<: AbstractCollection}) = Broadcast.DefaultArrayStyle{1}()

function Base.show(io::IO, c::AbstractCollection{rank}) where {rank}
    io = IOContext(io, :typeinfo => eltype(c))
    print(io, "<", length(c), " × ", eltype(c), ">[")
    join(io, [isassigned(c, i) ? sprint(show, c[i]; context=io) : "#undef" for i in eachindex(c)], ", ")
    print(io, "] with rank=$rank")
end


"""
    Collection(x, [Val(rank)])
"""
struct Collection{rank, T, V <: AbstractVector{T}} <: AbstractCollection{rank, T}
    parent::V
end

Collection{rank}(v::V) where {rank, T, V <: AbstractVector{T}} = Collection{rank, T, V}(v)
Collection(v) = Collection{1}(v)

"""
    collection(x, [Val(rank)])

Create collection with `x`.
"""
collection(v, ::Val{rank} = Val(1)) where {rank} = Collection{rank}(v)

Base.parent(c::Collection) = c.parent
Base.length(c::Collection) = length(parent(c))
@inline Base.getindex(c::Collection, i::Integer) = (@_propagate_inbounds_meta; parent(c)[i])
@inline Base.setindex!(c::Collection, v, i::Integer) = (@_propagate_inbounds_meta; parent(c)[i] = v)


"""
    LazyCollection
"""
struct LazyCollection{rank, Bc}
    bc::Bc
end

function LazyCollection{rank}(bc::Bc) where {rank, Bc}
    LazyCollection{rank, Bc}(bc)
end

Base.size(c::LazyCollection) = (length(c),)
Base.length(c::LazyCollection) = length(c.bc)
Base.getindex(c::LazyCollection, i) = (@_propagate_inbounds_meta; c.bc[i])

Broadcast.broadcastable(c::LazyCollection) = c.bc
Base.broadcasted(::typeof(identity), c::LazyCollection) = c.bc

Base.sum(c::LazyCollection) = sum(c.bc)
Base.collect(c::LazyCollection) = collect(c.bc)
Base.Array(c::LazyCollection) = copy(c.bc)


const UnionCollection{rank} = Union{AbstractCollection{rank}, LazyCollection{rank}}

Base.IndexStyle(::Type{<: Adjoint{<: Any, <: UnionCollection}}) = IndexLinear()
Base.size(c::Adjoint{<: Any, <: UnionCollection}) = (1, length(parent(c)))
@inline Base.getindex(c::Adjoint{<: Any, <: UnionCollection}, i::Integer) = (@_propagate_inbounds_meta; parent(c)[i])

for op in (:*, :/, :⋅, :⊗ , :×)
    @eval begin
        function Tensors.$op(c::UnionCollection{rank}, x) where {rank}
            LazyCollection{rank}(broadcasted($op, c, Ref(x)))
        end
        function Tensors.$op(x, c::UnionCollection{rank}) where {rank}
            LazyCollection{rank}(broadcasted($op, Ref(x), c))
        end
        function Tensors.$op(x::UnionCollection{0}, y::UnionCollection{0})
            LazyCollection{0}(broadcasted($op, x, Adjoint(y)))
        end
        function Tensors.$op(x::UnionCollection{rank}, y::UnionCollection{rank}) where {rank}
            LazyCollection{rank}(broadcasted($op, x, y))
        end
        function Tensors.$op(x::UnionCollection{L1}, y::UnionCollection{L2}) where {L1, L2}
            if L1 > L2
                LazyCollection{L1}(broadcasted($op, x, Ref(y)))
            else
                LazyCollection{L2}(broadcasted($op, Ref(x), y))
            end
        end
    end
end
