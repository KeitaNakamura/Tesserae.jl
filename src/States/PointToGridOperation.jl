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
        @assert ismatrix(x) == ismatrix(y)
        PointToGridOperation($op(x.gridvalues, y.gridvalues), ismatrix(x))
    end
end

# ∑ₚ(mₚ * vₚ * N) / mᵢ
for op in (:*, :/)
    @eval function Base.$op(x::PointToGridOperation, y::GridState)
        @assert !ismatrix(x)
        PointToGridOperation($op(x.gridvalues, GridStateCollection(y)), ismatrix(x))
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
