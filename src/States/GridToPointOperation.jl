struct GridToPointOperation{C <: AbstractCollection{2}} <: AbstractCollection{2}
    u_p::C
end

function ∑ᵢ(c::AbstractCollection{2})
    # Using `reduce` causes type instability in some cases
    GridToPointOperation(lazy(sum, c))
end

Base.length(x::GridToPointOperation) = length(x.u_p)
Base.getindex(x::GridToPointOperation, i::Int) = (@_propagate_inbounds_meta; x.u_p[i])

function set!(ps::PointState, x::GridToPointOperation)
    @inbounds for p in 1:length(ps)
        ps[p] = x.u_p[p]
    end
    ps
end

function Base.add_sum(a::ScalVec{dim, T}, b::ScalVec{dim, T})::ScalVec{dim, T} where {dim, T}
    ScalVec(a.x + b.x, a.∇x + b.∇x)
end
function Base.add_sum(a::VecTensor{dim, T, M}, b::VecTensor{dim, T, M})::VecTensor{dim, T, M} where {dim, T, M}
    VecTensor(a.x + b.x, a.∇x + b.∇x)
end
