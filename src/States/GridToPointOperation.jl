struct GridToPointOperation{C <: AbstractCollection{2}} <: AbstractCollection{2}
    u_p::C
end

function ∑ᵢ(c::AbstractCollection{2})
    GridToPointOperation(lazy(sum, c))
end

Base.length(x::GridToPointOperation) = length(x.u_p)
Base.getindex(x::GridToPointOperation, i::Int) = (@_propagate_inbounds_meta; x.u_p[i])
function Base.getindex(x::GridToPointOperation{<: LazyCollection{2}}, i::Int)
    @_propagate_inbounds_meta
    bc = x.u_p.bc
    args = Base.Broadcast._getindex(bc.args, i)
    Base.Broadcast._broadcast_getindex_evalf(bc.f, args...)
end

function Base.add_sum(a::ScalVec{dim, T}, b::ScalVec{dim, T})::ScalVec{dim, T} where {dim, T}
    ScalVec(a.x + b.x, a.∇x + b.∇x)
end
function Base.add_sum(a::VecTensor{dim, T, M}, b::VecTensor{dim, T, M})::VecTensor{dim, T, M} where {dim, T, M}
    VecTensor(a.x + b.x, a.∇x + b.∇x)
end
