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
# vector field
function add!(S::GridStateMatrix{Tensor{Tuple{dim, dim}}}, ∑ₚ∇N∇N::PointToGridMatrixOperation) where {dim}
    dofinds = S.dofindices
    @inbounds for p in eachindex(dofinds)
        mat = ∑ₚ∇N∇N[p]
        dofs = dofinds[p]
        for index in CartesianIndices(size(mat))
            i, j = Tuple(index)
            I = _compute_range(dofs[i], dim) # dof range
            J = _compute_range(dofs[j], dim) # dof range
            # if eltype of mat is scalar, create ScalarMatrix
            add!(S, _get_element(mat, index, dim), I, J)
        end
    end
    S
end
# scalar field
function add!(S::GridStateMatrix{<: Real}, ∑ₚ∇N∇N::PointToGridMatrixOperation)
    dofinds = S.dofindices
    @inbounds for p in eachindex(dofinds)
        mat = ∑ₚ∇N∇N[p]
        dofs = dofinds[p]
        add!(S, mat, dofs, dofs)
    end
    S
end

function set!(S::GridStateMatrix, list::List{PointToGridMatrixOperation})
    zeros!(S)
    for item in list
        add!(S, item)
    end
    S
end

set!(S::GridStateMatrix, x::ListGroup{PointToGridMatrixOperation}) = set!(S, List(x))


# for mass matrix
struct GridDiagonal{T} <: ListGroup{PointToGridMatrixOperation}
    parent::T
end
Base.parent(x::GridDiagonal) = x.parent

# vector field
function add!(S::GridStateMatrix{Tensor{Tuple{dim, dim}}}, mᵢᵢ::GridDiagonal{<: UnionGridState}) where {dim}
    # TODO: check if they have the same dofindices
    nzval = nonzeros(parent(mᵢᵢ))
    @inbounds for (dof, val) in enumerate(nzval)
        I = _compute_range(dof, dim)
        add!(S, FillArray(val, dim), I)
    end
    S
end
function add!(S::GridStateMatrix{Tensor{Tuple{dim, dim}}}, mᵢᵢ::GridDiagonal{<: PointToGridOperation}) where {dim, T}
    ∑ₚN = parent(mᵢᵢ)
    dofinds = S.dofindices
    @inbounds for p in eachindex(dofinds)
        N = ∑ₚN[p]
        dofs = dofinds[p]
        for i in eachindex(dofs)
            I = _compute_range(dofs[i], dim)
            add!(S, FillArray(N[i], dim), I)
        end
    end
    S
end

# scalar field
function add!(S::GridStateMatrix{<: Real}, mᵢᵢ::GridDiagonal{<: GridState})
    # TODO: check if they have the same dofindices
    nzval = nonzeros(parent(mᵢᵢ))
    add!(S, nzval, eachindex(nzval))
    S
end
function add!(S::GridStateMatrix{<: Real}, mᵢᵢ::GridDiagonal{<: PointToGridOperation})
    ∑ₚN = parent(mᵢᵢ)
    dofinds = S.dofindices
    @inbounds for p in eachindex(dofinds)
        N = ∑ₚN[p]
        dofs = dofinds[p]
        add!(S, N, dofs)
    end
    S
end
