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

Base.add_sum(a::ScalVec, b::ScalVec)::ScalVec = ScalVec(a.x + b.x, a.∇x + b.∇x)
Base.add_sum(a::VecTensor, b::VecTensor)::VecTensor = VecTensor(a.x + b.x, a.∇x + b.∇x)
