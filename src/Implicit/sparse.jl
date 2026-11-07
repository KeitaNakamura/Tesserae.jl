# -----------------------------------------------------------------------------
#  Sparse matrix utilities
# -----------------------------------------------------------------------------

const SparseMatrixCSCView{T, P <: SparseMatrixCSC} = SubArray{T, 2, P}

# `nonzeros` holds everything a fixed pattern can store, and zeroing it as a flat
# vector takes the memset path that `fill!` on the wrapper does not.
function fillzero!(matrix::AbstractSparseMatrix)
    fillzero!(nonzeros(matrix))
    matrix
end

function fillzero!(matrix::SparseMatrixCSCView)
    selected_rows, selected_cols = parentindices(matrix)
    selected_rows = issorted(selected_rows) ? selected_rows : sort(selected_rows)
    parent_matrix = parent(matrix)
    rows = rowvals(parent_matrix)
    values = nonzeros(parent_matrix)
    zero_value = zero_recursive(eltype(values))
    selected_stop = lastindex(selected_rows)
    @inbounds for col in selected_cols
        slots = nzrange(parent_matrix, col)
        isempty(slots) && continue
        selected_index = searchsortedfirst(selected_rows, rows[first(slots)])
        for slot in slots
            row = rows[slot]
            while selected_index ≤ selected_stop && selected_rows[selected_index] < row
                selected_index += 1
            end
            selected_index > selected_stop && break
            if selected_rows[selected_index] == row
                values[slot] = zero_value
            end
        end
    end
    matrix
end

# ---- parent matrix ----
#
# A wrapper keeping its values elsewhere, a view or a block of a shared matrix,
# overrides `matrix_parent`/`matrix_parent_indices` so that `extract` and the
# assemblers reach the parent and remap into it. `fillzero!` follows the same
# split: it must zero only the entries the target owns.
#
# What `mapped_index` returns is the caller's to interpret. Everywhere but one it
# maps a global logical DoF number to the parent's row or column. The exception is
# `add_entry_values!`, which passes a within-node DoF number and gets the parent's
# DoF *component* back; using that as an offset into `nonzeros` is valid only
# because `check_cartesian_sparse_matrix_view_indices` requires a view to select
# the same components at every node. `storageindex` is the one that really is a
# position in the stored values.

matrix_parent(matrix) = matrix
matrix_parent(matrix::Union{SparseMatrixCSCView, SparseMatrixBlockView, SparseMatrixBlocks}) = parent(matrix)

matrix_parent_indices(matrix) = axes(matrix)
matrix_parent_indices(matrix::Union{SparseMatrixCSCView, SparseMatrixBlockView}) = parentindices(matrix)
matrix_parent_indices(matrix::SparseMatrixBlocks) = axes(parent(matrix))

mapped_index(::Base.OneTo, index) = index
mapped_index(indices, index) = (@_propagate_inbounds_meta; indices[index])

# ---- sparse addition ----

function add!(A::SparseMatrixCSC, I::AbstractVector{Int}, J::AbstractVector{Int}, K::AbstractMatrix)
    if issorted(I)
        _add!(A, I, J, K, eachindex(I))
    else
        _add!(A, I, J, K, sortperm(I))
    end
end

function _add!(A::SparseMatrixCSC, I::AbstractVector{Int}, J::AbstractVector{Int}, K::AbstractMatrix, perm::AbstractVector{Int})
    @boundscheck checkbounds(A, I, J)
    @assert size(K) == map(length, (I, J))
    rows = rowvals(A)
    vals = nonzeros(A)
    @inbounds for j in eachindex(J)
        i = 1
        for k in nzrange(A, J[j])
            row = rows[k] # row candidate
            i′ = perm[i]
            if I[i′] == row
                vals[k] += K[i′,j]
                i += 1
                i > length(I) && break
            end
        end
        if i ≤ length(I) # some indices are not activated in sparse matrix `A`
            error("wrong sparsity pattern")
        end
    end
    A
end

function add!(A::AbstractMatrix, I::AbstractVector{Int}, J::AbstractVector{Int}, K::AbstractMatrix)
    @boundscheck checkbounds(A, I, J)
    @assert size(K) == map(length, (I, J))
    @inbounds @views A[I,J] .+= K
end
