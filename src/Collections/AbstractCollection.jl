"""
    AbstractCollection{rank}

Supertype for collections.
"""
abstract type AbstractCollection{rank} end

whichrank(::AbstractCollection{rank}) where {rank} = rank
whichrank(::Type{<: AbstractCollection{rank}}) where {rank} = rank
whichrank(::Any) = -100 # just use low value

Base.eltype(c::AbstractCollection) = typeof(first(c)) # try to getindex
# Above eltype throws error if collection is empty.
# `safe_eltype` returns `Union{}` for that case.
function safe_eltype(c::AbstractCollection)
    isempty(c) ? Union{} : eltype(c)
end

function Base.fill!(c::AbstractCollection, v)
    @simd for i in eachindex(c)
        @inbounds c[i] = v
    end
    c
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
@inline Base.getindex(c::AbstractCollection, I::Vararg{Any}) = (@_propagate_inbounds_meta; view(c, I...))

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
Broadcast.broadcastable(c::AbstractCollection) = error("AbstractCollection: Broadcast is not supported")

function set!(dest::Union{AbstractVector, AbstractCollection{rank}}, src::AbstractCollection{rank}) where {rank}
    @assert length(dest) == length(src)
    @simd for i in 1:length(dest)
        @inbounds dest[i] = src[i]
    end
    dest
end

# copied from Base
function Base.:(==)(A::Union{AbstractArray, AbstractCollection}, B::Union{AbstractArray, AbstractCollection})
    if axes(A) != axes(B)
        return false
    end
    anymissing = false
    for (a, b) in zip(A, B)
        eq = (a == b)
        if ismissing(eq)
            anymissing = true
        elseif !eq
            return false
        end
    end
    return anymissing ? missing : true
end

show_type_name(c::AbstractCollection) = typeof(c)
Base.summary(io::IO, c::AbstractCollection) =
    print(io, Base.dims2string(size(c)), " ", show_type_name(c), " with rank=", whichrank(c), ":")

function Base.show(io::IO, mime::MIME"text/plain", c::AbstractCollection)
    summary(io, c)
    println(io)
    Base.print_array(io, Array(c))
end

function Base.show(io::IO, c::AbstractCollection)
    print(io, collect(c))
end
