# -----------------------------------------------------------------------------
#  Sparse matrix utilities
# -----------------------------------------------------------------------------

const SparseMatrixCSCView{T, P <: SparseMatrixCSC} = SubArray{T, 2, P}

function fillzero!(matrix::SparseMatrixCSCView)
    selected_rows, selected_cols = parentindices(matrix)
    sorted_rows = issorted(selected_rows) ? selected_rows : sort(selected_rows)
    _fillzero_sparse_matrix_view!(parent(matrix), sorted_rows, selected_cols)
    matrix
end

function _fillzero_sparse_matrix_view!(matrix::SparseMatrixCSC, selected_rows, selected_cols)
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    zero_value = zero_recursive(eltype(values))
    selected_stop = lastindex(selected_rows)
    @inbounds for col in selected_cols
        slots = nzrange(matrix, col)
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
