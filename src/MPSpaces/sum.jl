struct SumToGrid{rank, Bc} <: AbstractCollection{rank, Any}
    collection::LazyCollection{rank, Bc}
end

function ∑ₚ(c::AbstractCollection{rank}) where {rank}
    SumToGrid(changerank(c, Val(rank))) # just to use identity for broadcast
end

function ∑ₚ(c::LazyCollection{rank}) where {rank}
    SumToGrid(c)
end

# Base.eltype(s::SumToGrid) = eltype(s.collection)
Base.length(s::SumToGrid) = length(s.collection)
Base.getindex(s::SumToGrid, p::Int) = (@_propagate_inbounds_meta; s.collection[p])

Broadcast.broadcastable(s::SumToGrid) = Broadcast.broadcastable(s.collection)
Base.Array(s::SumToGrid) = Array(s.collection)

Base.show(io::IO, s::SumToGrid) = print(io, "SumToGrid(", s.collection, ")")


struct SumToPoint{rank, Bc} <: AbstractCollection{rank, Any}
    collection::LazyCollection{rank, Bc}
end

function ∑ᵢ(c::AbstractCollection{rank}) where {rank}
    SumToPoint(changerank(c, Val(rank))) # just to use identity for broadcast
end

function ∑ᵢ(c::LazyCollection{rank}) where {rank}
    SumToPoint(c)
end

function ∑ᵢ(c::Union{AbstractCollection{1}, LazyCollection{1}})
    reduce(add, c)
end

# Base.eltype(s::SumToPoint) = eltype(s.collection)
Base.length(s::SumToPoint) = length(s.collection)
Base.getindex(s::SumToPoint{2}, i::Int) = (@_propagate_inbounds_meta; reduce(add, s.collection[i]))

add(a, b) = a + b
add(a::ScalarVector, b::ScalarVector) = ScalarVector(a.x + b.x, a.∇x + b.∇x)
add(a::VectorTensor, b::VectorTensor) = VectorTensor(a.x + b.x, a.∇x + b.∇x)
