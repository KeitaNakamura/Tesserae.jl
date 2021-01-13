struct PointToGridOperation{C <: AbstractCollection{2}}
    gridvalues::C
    ismatrix::Bool
end

gridvalues(c::PointToGridOperation, p::Int) = (@_propagate_inbounds_meta; c.gridvalues[p])
ismatrix(c::PointToGridOperation) = c.ismatrix

function ∑ₚ(c::AbstractCollection{2})
    ElType = eltype(c)
    if ElType <: AbstractCollection{0}
        return PointToGridOperation(c, false)
    elseif ElType <: AbstractCollection{-1}
        return PointToGridOperation(c, true)
    end
    throw(ArgumentError("wrong collection in ∑ₚ"))
end

for op in (:+, :-)
    @eval function Base.$op(x::PointToGridOperation, y::PointToGridOperation)
        PointToGridOperation($op(x.gridvalues, y.gridvalues))
    end
end

function set!(S::GridState, x::PointToGridOperation)
    @assert !ismatrix(x)
    nzval = nonzeros(zeros!(S))
    dofinds = S.dofindices
    @inbounds for p in eachindex(dofinds)
        u = view(nzval, dofinds[p])
        u .+= gridvalues(x, p)
    end
    S
end

_to_matrix(x::Real, dim::Int) = ScalarMatrix(x, dim, dim)
_to_matrix(x::Tensor{2}, dim::Int) = x
_get_element(mat, index, dim::Int) = (@_propagate_inbounds_meta; _to_matrix(mat.bc[index], dim))
_compute_range(dofs::Vector{Int}, i::Int, dim::Int) =
    (@_propagate_inbounds_meta; start = dim*(dofs[i]-1) + 1; start:start+dim-1)
function set!(S::GridStateMatrix{Vec{dim, T}}, x::PointToGridOperation) where {dim, T}
    @assert ismatrix(x)
    empty!(S)
    dofinds = S.dofindices
    @inbounds for p in eachindex(dofinds)
        mat = gridvalues(x, p)
        dofs = dofinds[p]
        for index in CartesianIndices(size(mat))
            i, j = Tuple(index)
            I = _compute_range(dofs, i, dim)
            J = _compute_range(dofs, j, dim)
            push!(S, _get_element(mat, index, dim), I, J)
        end
    end
    S
end

for op in (:*, :/)
    @eval function Base.$op(x::PointToGridOperation, y::GridState)
        PointToGridOperation($op(x.gridvalues, GridCollection(y)))
    end
end


struct GridToPointOperation{C <: AbstractCollection{2}} <: AbstractCollection{2}
    u_p::C
end

function ∑ᵢ(c::AbstractCollection{2})
    GridToPointOperation(lazy(reduce, add, c))
end

Base.length(x::GridToPointOperation) = length(x.u_p)
Base.getindex(x::GridToPointOperation, i::Int) = (@_propagate_inbounds_meta; x.u_p[i])

function set!(ps::PointState, x::GridToPointOperation)
    @inbounds for p in 1:length(ps)
        ps[p] = x.u_p[p]
    end
    ps
end

add(a, b) = a + b
add(a::ScalVec, b::ScalVec) = ScalVec(a.x + b.x, a.∇x + b.∇x)
add(a::VecTensor, b::VecTensor) = VecTensor(a.x + b.x, a.∇x + b.∇x)
