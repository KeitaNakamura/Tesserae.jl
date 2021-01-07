struct PointToGridOperation{C <: UnionCollection{2}}
    u_i::C
end

struct PointToGridMatrixOperation{C <: UnionCollection{2}}
    K_ij::C
end

function ∑ₚ(c::UnionCollection{2})
    ElType = eltype(c)
    if ElType <: UnionCollection{1}
        return PointToGridOperation(c)
    elseif ElType <: UnionCollection{-1}
        return PointToGridMatrixOperation(c)
    end
    throw(ArgumentError("wrong collection in ∑ₚ"))
end

for op in (:+, :-)
    @eval function Base.$op(x::PointToGridOperation, y::PointToGridOperation)
        PointToGridOperation($op(x.u_i, y.u_i))
    end
end

function set!(S::SparseArray, x::PointToGridOperation)
    nzval = nonzeros(zeros!(S))
    dofinds = S.dofindices
    @inbounds for p in eachindex(dofinds)
        u = view(nzval, dofinds[p])
        u .+= x.u_i[p]
    end
    S
end
