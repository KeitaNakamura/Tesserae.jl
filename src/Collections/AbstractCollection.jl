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
