# -----------------------------------------------------------------------------
#  Sparse matrix blocks
# -----------------------------------------------------------------------------

# ---- sparsity patterns ----

abstract type SparseMatrixPattern end

# This metadata proves the canonical Cartesian pattern without rescanning the
# parent CSC.
struct CartesianSparseMatrixPattern{N} <: SparseMatrixPattern
    mesh_size::Dims{N}
    sparsity_radius::Int
end

struct CellSparseMatrixPattern <: SparseMatrixPattern end

# ---- SparseMatrixBlockView ----

struct SparseMatrixBlockView{T, Ti, P <: SparseMatrixPattern} <: AbstractMatrix{T}
    matrix::SparseMatrixCSC{T, Ti}
    rows::UnitRange{Int}
    cols::UnitRange{Int}
    column_slots::Vector{UnitRange{Int}}
    pattern::P
end

Base.size(block::SparseMatrixBlockView) = (length(block.rows), length(block.cols))
Base.parent(block::SparseMatrixBlockView) = block.matrix
Base.parentindices(block::SparseMatrixBlockView) = (block.rows, block.cols)

function Base.getindex(block::SparseMatrixBlockView, i::Int, j::Int)
    @boundscheck checkbounds(block, i, j)
    @inbounds parent(block)[block.rows[i],block.cols[j]]
end

@inline function find_storageindex(block::SparseMatrixBlockView, i::Int, j::Int)
    @boundscheck checkbounds(block, i, j)

    matrix = parent(block)
    rows = rowvals(matrix)
    row = @inbounds block.rows[i]
    slots = @inbounds block.column_slots[j]
    isempty(slots) && return nothing
    slot = searchsortedfirst(rows, row, first(slots), last(slots), Base.Order.Forward)
    slot ∈ slots && rows[slot] == row ? slot : nothing
end

"""
    storageindex(block, i, j)

Return the index in `nonzeros(parent(block))` corresponding to the stored entry
`block[i,j]`. An `ArgumentError` is thrown when the entry is not part of the
fixed sparse pattern. Structural changes to the parent matrix invalidate
previously obtained storage indices.
"""
@inline function storageindex(block::SparseMatrixBlockView, i::Int, j::Int)
    slot = find_storageindex(block, i, j)
    isnothing(slot) && throw(ArgumentError("entry ($i, $j) is not stored in the sparse matrix block view"))
    slot
end

function Base.setindex!(block::SparseMatrixBlockView, value, i::Int, j::Int)
    slot = find_storageindex(block, i, j)
    if !isnothing(slot)
        @inbounds nonzeros(parent(block))[slot] = value
    elseif !iszero(value)
        throw(ArgumentError("cannot change the sparsity pattern of a sparse matrix block view"))
    end
    block
end

function fillzero!(block::SparseMatrixBlockView)
    values = nonzeros(parent(block))
    zero_value = zero_recursive(eltype(values))
    for slots in block.column_slots, slot in slots
        @inbounds values[slot] = zero_value
    end
    block
end

SparseArrays.nnz(block::SparseMatrixBlockView) = sum(length, block.column_slots)

# ---- SparseMatrixBlocks ----

# Owns the parent CSC and creates block views from shared offsets and slot tables.
struct SparseMatrixBlocks{T, Ti, P <: SparseMatrixPattern} <: AbstractMatrix{SparseMatrixBlockView{T, Ti, P}}
    matrix::SparseMatrixCSC{T, Ti}
    field_offsets::Vector{Int}
    column_slots::Matrix{Vector{UnitRange{Int}}}
    pattern::P
end

Base.size(blocks::SparseMatrixBlocks) = size(blocks.column_slots)
Base.parent(blocks::SparseMatrixBlocks) = blocks.matrix

function Base.getindex(blocks::SparseMatrixBlocks, i::Int, j::Int)
    @boundscheck checkbounds(blocks, i, j)
    @inbounds SparseMatrixBlockView(
        blocks.matrix,
        (blocks.field_offsets[i] + 1):blocks.field_offsets[i + 1],
        (blocks.field_offsets[j] + 1):blocks.field_offsets[j + 1],
        blocks.column_slots[i,j],
        blocks.pattern,
    )
end

function fillzero!(blocks::SparseMatrixBlocks)
    fillzero!(parent(blocks))
    blocks
end

# ---- construction ----

# -- field matrix API --

"""
    create_block_sparse_matrix(mesh; ndofs)
    create_block_sparse_matrix(meshes; ndofs)
    create_block_sparse_matrix(basis, mesh; ndofs)

Create a monolithic sparse matrix for multiple fields and return its block
views. `ndofs` is a tuple containing the number of DoFs per node for each
field. A tuple of meshes assigns one mesh to each field. For Cartesian meshes,
pass the basis explicitly.

```julia
blocks = create_block_sparse_matrix((velocity_mesh, pressure_mesh); ndofs=(2, 1))
K = parent(blocks)
Kuu, Kup = blocks[1,1], blocks[1,2]
Kpu, Kpp = blocks[2,1], blocks[2,2]
```

The parent CSC uses the sparsity pattern generated from the supplied
discretization. Block views share its fixed structure and permit updates only
at stored positions. Structural changes made through `parent(blocks)`
invalidate all block views.
"""
function create_block_sparse_matrix end

create_block_sparse_matrix(basis::Basis, mesh::CartesianMesh; ndofs::NTuple{N, Int}) where {N} = create_block_sparse_matrix(Float64, basis, mesh; ndofs)
create_block_sparse_matrix(::Type{T}, basis::Basis, mesh::CartesianMesh; ndofs::NTuple{N, Int}) where {T, N} = _create_block_sparse_matrix(T, basis, mesh, ndofs)

create_block_sparse_matrix(mesh::Union{FEMesh, IGAMesh}; ndofs::NTuple{N, Int}) where {N} = create_block_sparse_matrix(Float64, mesh; ndofs)
function create_block_sparse_matrix(::Type{T}, mesh::Union{FEMesh, IGAMesh}; ndofs::NTuple{N, Int}) where {T, N}
    meshes = ntuple(_ -> mesh, N)
    _create_block_sparse_matrix(T, meshes, ndofs)
end

create_block_sparse_matrix(meshes::Union{NTuple{N, FEMesh}, NTuple{N, IGAMesh}}; ndofs::NTuple{N, Int}) where {N} = create_block_sparse_matrix(Float64, meshes; ndofs)
create_block_sparse_matrix(::Type{T}, meshes::Union{NTuple{N, FEMesh}, NTuple{N, IGAMesh}}; ndofs::NTuple{N, Int}) where {T, N} = _create_block_sparse_matrix(T, meshes, ndofs)

# -- helpers --

function check_field_ndofs(ndofs)
    isempty(ndofs) && throw(ArgumentError("at least one field is required"))
    all(>(0), ndofs) || throw(ArgumentError("field DoF counts must be positive"))
    nothing
end

function _create_sparse_matrix_blocks(::Type{T}, I, J, field_offsets, pattern::SparseMatrixPattern) where {T}
    matrix_size = last(field_offsets)
    matrix = sparse(I, J, zeros(T, length(I)), matrix_size, matrix_size)
    nfields = length(field_offsets) - 1
    column_slots = [Vector{UnitRange{Int}}(undef, field_offsets[j + 1] - field_offsets[j]) for i in 1:nfields, j in 1:nfields]
    rows = rowvals(matrix)

    # Rows are sorted within each CSC column, so each field occupies one contiguous slot range.
    for j in 1:nfields, local_col in 1:(field_offsets[j + 1] - field_offsets[j])
        slots = nzrange(matrix, field_offsets[j] + local_col)
        slot = first(slots)
        stop = last(slots) + 1
        for i in 1:nfields
            first_slot = slot
            @inbounds while slot < stop && rows[slot] ≤ field_offsets[i + 1]
                slot += 1
            end
            column_slots[i,j][local_col] = first_slot:(slot - 1)
        end
    end

    SparseMatrixBlocks(matrix, field_offsets, column_slots, pattern)
end

# -- MPM --

function _create_block_sparse_matrix(::Type{T}, basis::Basis, mesh::CartesianMesh, ndofs::NTuple{N, Int}) where {T, N}
    check_field_ndofs(ndofs)
    field_sizes = [ndof * length(mesh) for ndof in ndofs]
    field_offsets = cumsum([0; field_sizes])
    I, J = Int[], Int[]
    for j in eachindex(ndofs), i in eachindex(ndofs)
        _append_sparse_pattern!(I, J, field_offsets[i], field_offsets[j], basis, mesh, ndofs[i], ndofs[j])
    end
    pattern = CartesianSparseMatrixPattern(size(mesh), support_width(basis) - 1)
    _create_sparse_matrix_blocks(T, I, J, field_offsets, pattern)
end

# -- FEM and IGA --

function _create_block_sparse_matrix(::Type{T}, meshes::Union{NTuple{N, FEMesh}, NTuple{N, IGAMesh}}, ndofs::NTuple{N, Int}) where {T, N}
    check_field_ndofs(ndofs)
    field_sizes = [ndofs[i] * length(meshes[i]) for i in eachindex(meshes)]
    field_offsets = cumsum([0; field_sizes])
    I, J = Int[], Int[]
    for j in eachindex(meshes), i in eachindex(meshes)
        _append_sparse_pattern!(I, J, field_offsets[i], field_offsets[j], meshes[i], meshes[j], ndofs[i], ndofs[j])
    end
    _create_sparse_matrix_blocks(T, I, J, field_offsets, CellSparseMatrixPattern())
end

