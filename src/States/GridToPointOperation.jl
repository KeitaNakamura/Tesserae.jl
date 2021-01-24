struct GridToPointOperation{C <: AbstractCollection{2}} <: AbstractCollection{2}
    u_p::C
end

function ∑ᵢ(c::AbstractCollection{2})
    # TODO: fix type instability
    GridToPointOperation(lazy(sum, c))
end

Base.length(x::GridToPointOperation) = length(x.u_p)
Base.getindex(x::GridToPointOperation, i::Int) = (@_propagate_inbounds_meta; x.u_p[i])

function Base.add_sum(a::ScalVec{dim, T}, b::ScalVec{dim, T})::ScalVec{dim, T} where {dim, T}
    ScalVec(a.x + b.x, a.∇x + b.∇x)
end
function Base.add_sum(a::VecTensor{dim, T, M}, b::VecTensor{dim, T, M})::VecTensor{dim, T, M} where {dim, T, M}
    VecTensor(a.x + b.x, a.∇x + b.∇x)
end
