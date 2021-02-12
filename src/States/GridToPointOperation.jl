struct GridToPointOperation{C <: AbstractCollection{2}} <: AbstractCollection{2}
    u_p::C
end

function ∑ᵢ(c::AbstractCollection{2})
    GridToPointOperation(lazy(_sum, c))
end

Base.length(x::GridToPointOperation) = length(x.u_p)
Base.getindex(x::GridToPointOperation, i::Int) = (@_propagate_inbounds_meta; x.u_p[i])
# function Base.getindex(x::GridToPointOperation{<: LazyCollection{2}}, i::Int)
    # @_propagate_inbounds_meta
    # c = x.u_p
    # args = Collections._getindex((), c.args, i)
    # c.f(args...)
# end

function _sum(c)
    x = first(c)
    @simd for i in 2:length(c)
        @inbounds x = add(x, c[i])
    end
    x
end

add(a, b) = a + b
function add(a::ScalVec{dim, T}, b::ScalVec{dim, T})::ScalVec{dim, T} where {dim, T}
    ScalVec(a.x + b.x, a.∇x + b.∇x)
end
function add(a::VecTensor{dim, T, M}, b::VecTensor{dim, T, M})::VecTensor{dim, T, M} where {dim, T, M}
    VecTensor(a.x + b.x, a.∇x + b.∇x)
end
