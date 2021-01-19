struct PointToGridOperation{C <: AbstractCollection{2}}
    parent::C
end

Base.parent(c::PointToGridOperation) = c.parent
Base.length(c::PointToGridOperation) = length(parent(c))
Base.getindex(c::PointToGridOperation, p::Int) = (@_propagate_inbounds_meta; parent(c)[p])

∑ₚ(c::AbstractCollection{2}) = PointToGridOperation(c)
∑ₚ(c::AbstractCollection{rank}) where {rank} = throw(ArgumentError("cannot apply ∑ₚ(...) for rank=$rank collection."))

for op in (:+, :-)
    @eval function Base.$op(x::PointToGridOperation, y::PointToGridOperation)
        PointToGridOperation($op(parent(x), parent(y)))
    end
end

for op in (:*, :/)
    @eval begin
        if $op == /
            # ∑ₚ(mₚ * vₚ * N) / mᵢ
            function Base.$op(x::PointToGridOperation, y::GridState)
                PointToGridOperation($op(parent(x), GridStateCollection(y)))
            end
        end
        # wrong calculation
        Base.$op(x::PointToGridOperation, y::AbstractCollection) =
            throw(ArgumentError("∑ₚ(...) cannot be computed with other collections."))
        Base.$op(x::AbstractCollection, y::PointToGridOperation) =
            throw(ArgumentError("∑ₚ(...) cannot be computed with other collections."))
    end
end

function set!(S::GridState, ∑ₚN::PointToGridOperation)
    nzval = nonzeros(zeros!(S))
    dofinds = S.dofindices
    @inbounds for p in eachindex(dofinds)
        u = view(nzval, dofinds[p])
        u .+= ∑ₚN[p]
    end
    S
end

_to_matrix(x::Real, dim::Int) = ScalarMatrix(x, dim, dim)
_to_matrix(x::SecondOrderTensor, dim::Int) = x
_get_element(mat, index, dim::Int) = (@_propagate_inbounds_meta; _to_matrix(mat.bc[index], dim))
_compute_range(dofs::Vector{Int}, i::Int, dim::Int) =
    (@_propagate_inbounds_meta; start = dim*(dofs[i]-1) + 1; start:start+dim-1)
function set!(S::GridStateMatrix{Vec{dim, T}}, ∑ₚ∇N∇N::PointToGridOperation) where {dim, T}
    empty!(S)
    dofinds = S.dofindices
    @inbounds for p in eachindex(dofinds)
        mat = ∑ₚ∇N∇N[p]
        dofs = dofinds[p]
        for index in CartesianIndices(size(mat))
            i, j = Tuple(index)
            I = _compute_range(dofs, i, dim) # dof range
            J = _compute_range(dofs, j, dim) # dof range
            # if eltype of mat is scalar, create ScalarMatrix
            push!(S, _get_element(mat, index, dim), I, J)
        end
    end
    S
end
