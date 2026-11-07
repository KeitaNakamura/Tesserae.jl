# Active degree-of-freedom extraction

# ---- extraction ----

"""
    extract(matrix, dofmap_row, dofmap_col = dofmap_row)

Extract the active degrees of freedom of a matrix.
"""
function extract(matrix::AbstractMatrix, dofmap_i, dofmap_j = dofmap_i)
    I, J = _parent_indices_for_extract(matrix, dofmap_i, dofmap_j)
    matrix_parent(matrix)[I, J]
end
function extract(::typeof(view), matrix::AbstractMatrix, dofmap_i, dofmap_j = dofmap_i)
    I, J = _parent_indices_for_extract(matrix, dofmap_i, dofmap_j)
    view(matrix_parent(matrix), I, J)
end

function _parent_indices_for_extract(matrix::AbstractMatrix, dofmap_i::Union{AbstractDofMap, Colon}, dofmap_j::Union{AbstractDofMap, Colon})
    check_for_extract(matrix, dofmap_i, dofmap_j)
    device = dof_index_device(matrix)
    row_parent_indices, col_parent_indices = matrix_parent_indices(matrix)
    parent_matrix = matrix_parent(matrix)
    extract_dofs(device, row_parent_indices, size(parent_matrix, 1), dofmap_i),
    extract_dofs(device, col_parent_indices, size(parent_matrix, 2), dofmap_j)
end

function check_for_extract(matrix::AbstractMatrix, dofmap_i, dofmap_j)
    check_dofmap_size(size(matrix, 1), dofmap_i)
    check_dofmap_size(size(matrix, 2), dofmap_j)
end

function check_for_extract(blocks::SparseMatrixBlocks, dofmap_i, dofmap_j)
    check_block_dofmap(blocks, dofmap_i)
    check_block_dofmap(blocks, dofmap_j)
end

# A device sparse matrix has no `getindex` for a vector of DoF numbers, so the
# reduction goes through the Boolean mask the map was built from, embedded into
# the parent's axis when the target is a block or a range view of one. Nothing
# else needs the distinction, and asking a generic `AbstractMatrix` where its
# values live is not answerable.
dof_index_device(matrix) = _dof_index_device(matrix_parent(matrix))
_dof_index_device(parent_matrix) = CPUDevice()
_dof_index_device(parent_matrix::AbstractSparseMatrix) = get_device(parent_matrix)

extract_dofs(::AbstractDevice, axis_indices, axis_length, ::Colon) = axis_indices
extract_dofs(::CPUDevice, axis_indices, axis_length, dofmap::AbstractDofMap) = mapped_index(axis_indices, dofs(dofmap))
function extract_dofs(device::GPUDevice, axis_indices, axis_length, dofmap::AbstractDofMap)
    check_dofmap_device(device, dofmap)
    embed_axis_mask(axis_indices, axis_length, dof_mask(dofmap))
end

@noinline function check_dofmap_device(device, dofmap)
    get_device(dof_mask(dofmap)) === device ||
        throw(ArgumentError("extract: the DoF map and the matrix must be on the same device; build the map from a mask living where the matrix does"))
    nothing
end

embed_axis_mask(::Base.OneTo, axis_length, mask) = mask
function embed_axis_mask(axis_indices::UnitRange{Int}, axis_length, mask)
    padded = fillzero!(similar(mask, axis_length))
    copyto!(view(padded, axis_indices), mask)
    padded
end
@noinline embed_axis_mask(axis_indices, axis_length, mask) =
    throw(ArgumentError("extract: on GPU the target's rows and columns must be contiguous ranges of its parent"))

function check_dofmap_size(matrix_size::Integer, dofmap::AbstractDofMap)
    matrix_size == full_ndofs(dofmap) || throw(DimensionMismatch("matrix and DoF map sizes must match"))
    nothing
end
check_dofmap_size(::Integer, ::Colon) = nothing

function check_block_dofmap(blocks::SparseMatrixBlocks, dofmap::BlockDofMap)
    length(dofmap) == size(blocks, 1) || throw(DimensionMismatch("matrix and DoF map must have the same number of blocks"))
    for i in eachindex(dofmap.maps)
        block_size = blocks.field_offsets[i + 1] - blocks.field_offsets[i]
        full_ndofs(dofmap[i]) == block_size || throw(DimensionMismatch("matrix and DoF map block sizes must match"))
    end
    nothing
end
function check_block_dofmap(::SparseMatrixBlocks, ::DofMap)
    throw(ArgumentError("extracting a block matrix requires one DoF map per block"))
end
check_block_dofmap(::SparseMatrixBlocks, ::Colon) = nothing

# ---- value-only extraction ----

"""
    extract!(dest, matrix, dofmap_row, dofmap_col = dofmap_row)

Refill `dest` with the active entries of `matrix`, leaving its sparsity pattern
untouched. `dest` must come from [`extract`](@ref) with the same DoF maps, whose
masks must not have changed since.

For a matrix view, the selected parent rows must be a contiguous range.
On GPU the selected parent columns must also be contiguous.

Assembling into `matrix` and reducing it once per Newton iteration is what this
is for: [`extract`](@ref) allocates a matrix and derives its pattern, while this
walks the stored values only.
"""
function extract!(dest::AbstractSparseMatrix, matrix::AbstractMatrix, dofmap_i::AbstractDofMap, dofmap_j::AbstractDofMap = dofmap_i)
    check_for_extract(matrix, dofmap_i, dofmap_j)
    device = dof_index_device(matrix)
    size(dest) == (ndofs(dofmap_i), ndofs(dofmap_j)) ||
        throw(DimensionMismatch("`dest` and the active degrees of freedom must have the same size"))
    get_device(dest) === device || throw(ArgumentError("`dest` and the matrix must be on the same device"))
    if device isa GPUDevice
        check_dofmap_device(device, dofmap_i)
        check_dofmap_device(device, dofmap_j)
    end
    row_parent_indices, col_parent_indices = matrix_parent_indices(matrix)
    parent_matrix = matrix_parent(matrix)
    _extract_values!(device, nonzeros(dest), SparseArrays.getcolptr(dest),
                     dofs(dofmap_j), dof_mask(dofmap_i),
                     extract_row_offset(row_parent_indices), column_slots_of(matrix), col_parent_indices,
                     SparseArrays.getcolptr(parent_matrix), rowvals(parent_matrix), nonzeros(parent_matrix))
    dest
end

extract_row_offset(indices::Base.OneTo) = 0
extract_row_offset(indices::UnitRange{Int}) = first(indices) - 1
@noinline extract_row_offset(indices) =
    throw(ArgumentError("extract!: the target's rows must be a contiguous range of its parent"))

# A block view walks the slots already restricted to its rows; anything else walks
# the parent column and filters by the row window.
column_slots_of(matrix) = nothing
column_slots_of(block::SparseMatrixBlockView) = block.column_slots

@inline function source_column_slots(src_colptr, ::Nothing, col_parent_indices, jd)
    @_propagate_inbounds_meta
    src_col = mapped_index(col_parent_indices, jd)
    src_colptr[src_col]:(src_colptr[src_col+1]-1)
end
@inline function source_column_slots(src_colptr, column_slots, col_parent_indices, jd)
    @_propagate_inbounds_meta
    column_slots[jd]
end

# Both destination and source keep their rows sorted within a column, so the kept
# entries of one parent column arrive in the destination's own order and only the
# membership test is needed.
@inline function extract_column_values!(dest_values, dest_colptr, slots, rowmask, row_offset, src_rows, src_values, col)
    @inbounds begin
        slot = dest_colptr[col]
        for src_slot in slots
            row = src_rows[src_slot] - row_offset
            if 1 <= row <= length(rowmask) && rowmask[row]
                dest_values[slot] = src_values[src_slot]
                slot += 1
            end
        end
    end
end

@kernel function gpukernel_extract_values!(dest_values, dest_colptr, @Const(columns), @Const(rowmask), row_offset, column_slots, col_parent_indices, @Const(src_colptr), @Const(src_rows), @Const(src_values))
    col = @index(Global)
    @inbounds slots = source_column_slots(src_colptr, column_slots, col_parent_indices, columns[col])
    extract_column_values!(dest_values, dest_colptr, slots, rowmask, row_offset, src_rows, src_values, col)
end

function _extract_values!(::CPUDevice, dest_values, dest_colptr, columns, rowmask, row_offset, column_slots, col_parent_indices, src_colptr, src_rows, src_values)
    @inbounds for col in eachindex(columns)
        slots = source_column_slots(src_colptr, column_slots, col_parent_indices, columns[col])
        extract_column_values!(dest_values, dest_colptr, slots, rowmask, row_offset, src_rows, src_values, col)
    end
    nothing
end

function _extract_values!(device::GPUDevice, dest_values, dest_colptr, columns, rowmask, row_offset, column_slots, col_parent_indices, src_colptr, src_rows, src_values)
    kernel = gpukernel_extract_values!(get_backend(device))
    kernel(dest_values, dest_colptr, columns, rowmask, row_offset, column_slots, col_parent_indices, src_colptr, src_rows, src_values; ndrange=length(columns))
    nothing
end
