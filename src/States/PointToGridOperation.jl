struct PointToGridOperation{C <: AbstractCollection{2}} <: ListGroup{PointToGridOperation}
    parent::C
end
Base.parent(c::PointToGridOperation) = c.parent
Base.getindex(c::PointToGridOperation, p::Int) = (@_propagate_inbounds_meta; parent(c)[p])

struct PointToGridMatrixOperation{C <: AbstractCollection{2}} <: ListGroup{PointToGridMatrixOperation}
    parent::C
end
Base.parent(c::PointToGridMatrixOperation) = c.parent
Base.getindex(c::PointToGridMatrixOperation, p::Int) = (@_propagate_inbounds_meta; parent(c)[p])

function ∑ₚ(c::AbstractCollection{2})
    x = first(c)
    if x isa AbstractCollection{0} || x isa AbstractCollection{1}
        # shape value `N` is used only one time or not used at all
        return PointToGridOperation(c)
    elseif x isa AbstractCollection{-1} && ndims(x) == 2
        # shape value `N` is used two times
        return PointToGridMatrixOperation(c)
    end
    throw(ArgumentError("wrong collection in ∑ₚ(...)"))
end
∑ₚ(c::AbstractCollection{rank}) where {rank} = throw(ArgumentError("∑ₚ(...) can be applied only for rank=2 collections, got rank=$rank."))

# PointToGridOperation
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

function add!(S::GridState, ∑ₚN::PointToGridOperation)
    nzval = nonzeros(S)
    dofinds = S.dofindices
    @inbounds for p in eachindex(dofinds)
        u = view(nzval, dofinds[p])
        u .+= ∑ₚN[p]
    end
    S
end

function set!(S::GridState, list::List{PointToGridOperation})
    zeros!(S)
    for item in list
        add!(S, item)
    end
    S
end

set!(S::GridState, x::ListGroup{PointToGridOperation}) = set!(S, List(x))

_to_matrix(x::Real, dim::Int) = ScalarMatrix(x, dim, dim)
_to_matrix(x::SecondOrderTensor, dim::Int) = x
_get_element(mat, index, dim::Int) = (@_propagate_inbounds_meta; _to_matrix(mat.bc[index], dim))
_compute_range(dof::Int, dim::Int) = (start = dim*(dof-1) + 1; start:start+dim-1)
function add!(S::GridStateMatrix{Vec{dim, T}}, ∑ₚ∇N∇N::PointToGridMatrixOperation) where {dim, T}
    dofinds = S.dofindices
    @inbounds for p in eachindex(dofinds)
        mat = ∑ₚ∇N∇N[p]
        dofs = dofinds[p]
        for index in CartesianIndices(size(mat))
            i, j = Tuple(index)
            I = _compute_range(dofs[i], dim) # dof range
            J = _compute_range(dofs[j], dim) # dof range
            # if eltype of mat is scalar, create ScalarMatrix
            push!(S, _get_element(mat, index, dim), I, J)
        end
    end
    S
end

function set!(S::GridStateMatrix{Vec{dim, T}}, list::List{PointToGridMatrixOperation}) where {dim, T}
    empty!(S)
    for item in list
        add!(S, item)
    end
end

set!(S::GridStateMatrix, x::ListGroup{PointToGridMatrixOperation}) = set!(S, List(x))
