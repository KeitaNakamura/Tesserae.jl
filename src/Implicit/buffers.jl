# -----------------------------------------------------------------------------
#  LocalMatrixBuffer
# -----------------------------------------------------------------------------

# ---- construction ----

struct LocalMatrixBuffer{M <: Matrix, I, J}
    matrix::M
    row_ndofs::Int
    col_ndofs::Int
    row_nodes::I
    col_nodes::J
end

# Only the cell path reads the cache, but the macro binds it in every expansion
# and the transfer closure captures it either way. A `TaskLocalValue` there makes
# the closure non-isbits, which a GPU launch rejects.
local_matrix_cache(matrix, dof_table_i, weights_i, dof_table_j, weights_j) = nothing

function local_matrix_cache(matrix, dof_table_i, weights_i::BasisWeightArray{<:Any, <:Any, <:CellSupportMatrix}, dof_table_j, weights_j::BasisWeightArray{<:Any, <:Any, <:CellSupportMatrix})
    T = eltype(matrix)
    TaskLocalValue{Matrix{T}}() do
        row_size = size(dof_table_i, 1) * nsupportnodes(basis(weights_i))
        col_size = size(dof_table_j, 1) * nsupportnodes(basis(weights_j))
        Matrix{T}(undef, row_size, col_size)
    end
end

@inline function local_matrix_buffer(cache::TaskLocalValue{<:Matrix}, row_dof_table, row_nodes, col_dof_table, col_nodes)
    row_ndofs = size(row_dof_table, 1)
    col_ndofs = size(col_dof_table, 1)
    dims = row_ndofs * length(row_nodes), col_ndofs * length(col_nodes)
    matrix = cache[]
    # Validate reuse of the task-local allocation before the inbounds entry loop.
    size(matrix) == dims || throw(DimensionMismatch("local matrix size changed during assembly"))
    # Keep the cell support for the final scatter of the local matrix.
    LocalMatrixBuffer(matrix, row_ndofs, col_ndofs, row_nodes, col_nodes)
end

# ---- DoF indexing ----

local_dofs(ndofs::Int, index::Integer) = node_dofs(index, ndofs)

# ---- assembly ----

function assemble_first!(buffer::LocalMatrixBuffer, ip, jp, value)
    @_propagate_inbounds_meta
    buffer.matrix[local_dofs(buffer.row_ndofs, ip), local_dofs(buffer.col_ndofs, jp)] .= value
    buffer
end

function assemble_add!(buffer::LocalMatrixBuffer, ip, jp, value)
    @_propagate_inbounds_meta
    @views buffer.matrix[local_dofs(buffer.row_ndofs, ip), local_dofs(buffer.col_ndofs, jp)] .+= value
    buffer
end

function finish_assembly!(assembler, buffer::LocalMatrixBuffer, orientation)
    @_propagate_inbounds_meta
    row_nodes, col_nodes = orientation((buffer.row_nodes, buffer.col_nodes))
    add!(assembler, row_nodes, col_nodes, orient_matrix_entry(orientation, buffer.matrix))
end

# -----------------------------------------------------------------------------
#  BlockMatrixBuffer
# -----------------------------------------------------------------------------

# ---- construction ----

struct BlockAssembly{I <: CartesianIndices}
    nodes_i::I
    nodes_j::I
    matrix_buffer_pool::BlockMatrixBufferPool
end

struct BlockMatrixBufferKey{T, N}
    row_size::Dims{N}
    col_size::Dims{N}
    col_offset::CartesianIndex{N}
    row_ndofs::Int
    col_ndofs::Int
    sparsity_radius::Int
end

struct BlockMatrixBuffer{T, N}
    values::Vector{T}
    node_colstarts::Vector{Int}
    key::BlockMatrixBufferKey{T, N}
end

function BlockMatrixBufferKey(assembler::CartesianSparseMatrixAssembler, row_nodes::CartesianIndices{N}, col_nodes::CartesianIndices{N}) where {N}
    (; matrix, row_dof_table, col_dof_table, sparsity_radius) = assembler
    BlockMatrixBufferKey{eltype(matrix), N}(
        size(row_nodes),
        size(col_nodes),
        first(col_nodes) - first(row_nodes),
        size(row_dof_table, 1),
        size(col_dof_table, 1),
        sparsity_radius,
    )
end

function BlockMatrixBuffer(key::BlockMatrixBufferKey{T, N}) where {T, N}
    (; row_size, col_size, col_offset, row_ndofs, col_ndofs, sparsity_radius) = key
    node_colstarts = Vector{Int}(undef, prod(col_size))
    slot = 1
    for (col, col_node) in enumerate(CartesianIndices(col_size))
        node_colstarts[col] = slot
        row_nodes = cartesian_neighbor_nodes(col_node + col_offset, row_size, sparsity_radius)
        slot += row_ndofs * col_ndofs * length(row_nodes)
    end
    BlockMatrixBuffer(zeros(T, slot - 1), node_colstarts, key)
end

# ---- buffer pool ----

function acquire!(pool::BlockMatrixBufferPool, key::BlockMatrixBufferKey{T, N}) where {T, N}
    buffer = lock(pool.lock) do
        buffers = get!(Vector{Any}, pool.buffers, key)
        isempty(buffers) ? nothing : pop!(buffers)
    end
    buffer === nothing && return BlockMatrixBuffer(key)
    buffer = buffer::BlockMatrixBuffer{T, N}
    fillzero!(buffer.values)
    buffer
end

function release!(pool::BlockMatrixBufferPool, buffer::BlockMatrixBuffer)
    lock(pool.lock) do
        push!(get!(Vector{Any}, pool.buffers, buffer.key), buffer)
    end
    nothing
end

# ---- block buffer selection ----

# Cartesian blocks share CSC columns and always use a buffer. Generic assembly
# writes directly into the disjoint node sets of the block schedule.
function block_matrix_buffer(assembler::CartesianSparseMatrixAssembler, pool::BlockMatrixBufferPool, row_nodes, col_nodes)
    acquire!(pool, BlockMatrixBufferKey(assembler, row_nodes, col_nodes))
end

block_matrix_buffer(::GenericMatrixAssembler, ::BlockMatrixBufferPool, row_nodes, col_nodes) = nothing

# ---- assembly ----

# -- accumulation --

function add_entry!(buffer::BlockMatrixBuffer{T, N}, row_nodes::CartesianIndices{N}, col_nodes::CartesianIndices{N}, row_node::CartesianIndex{N}, col_node::CartesianIndex{N}, value) where {T, N}
    @_propagate_inbounds_meta
    @boundscheck check_block_matrix_entry(buffer, row_nodes, col_nodes, row_node, col_node, value)

    (; values, node_colstarts, key) = buffer
    (; row_size, col_size, col_offset, row_ndofs, col_ndofs, sparsity_radius) = key

    local_row_node = row_node - first(row_nodes) + oneunit(row_node)
    local_col_node = col_node - first(col_nodes) + oneunit(col_node)
    neighboring_rows = cartesian_neighbor_nodes(local_col_node + col_offset, row_size, sparsity_radius)
    local_col = LinearIndices(col_size)[local_col_node]

    slot = node_colstarts[local_col] + cartesian_slot_offset(local_row_node, neighboring_rows, row_ndofs * col_ndofs)
    add_entry_values!(SerialScatter(), values, slot, value)

    buffer
end

@noinline function check_block_matrix_entry(buffer::BlockMatrixBuffer{T, N}, row_nodes::CartesianIndices{N}, col_nodes::CartesianIndices{N}, row_node::CartesianIndex{N}, col_node::CartesianIndex{N}, value) where {T, N}
    (; key) = buffer
    (; row_size, col_size, col_offset, row_ndofs, col_ndofs, sparsity_radius) = key
    size(row_nodes) == row_size || throw(DimensionMismatch("row support size does not match block matrix buffer"))
    size(col_nodes) == col_size || throw(DimensionMismatch("column support size does not match block matrix buffer"))
    first(col_nodes) - first(row_nodes) == col_offset || throw(DimensionMismatch("support offset does not match block matrix buffer"))
    row_node ∈ row_nodes || throw(BoundsError(row_nodes, row_node))
    col_node ∈ col_nodes || throw(BoundsError(col_nodes, col_node))
    check_matrix_entry_size(value, row_ndofs, col_ndofs)
    local_row_node = row_node - first(row_nodes) + oneunit(row_node)
    local_col_node = col_node - first(col_nodes) + oneunit(col_node)
    neighboring_rows = cartesian_neighbor_nodes(local_col_node + col_offset, row_size, sparsity_radius)
    local_row_node ∈ neighboring_rows || throw(ArgumentError("matrix entry is outside the block sparsity pattern"))
    nothing
end

# -- scatter --

function scatter!(assembler::CartesianSparseMatrixAssembler, buffer::BlockMatrixBuffer{T, N}, row_nodes::CartesianIndices{N}, col_nodes::CartesianIndices{N}) where {T, N}
    @_propagate_inbounds_meta
    @boundscheck check_block_matrix_scatter(assembler, buffer, row_nodes, col_nodes)

    (; matrix, row_dof_table, row_slots_per_node, sparsity_radius) = assembler
    parent_matrix = matrix_parent(matrix)
    row_components = cartesian_matrix_row_components(assembler)
    (; values, node_colstarts, key) = buffer
    (; row_size, col_size, col_offset, row_ndofs, col_ndofs) = key

    mesh_size = Base.tail(size(row_dof_table))
    matrix_values = nonzeros(parent_matrix)
    first_row_node = first(row_nodes)
    first_col_node = first(col_nodes)
    for (local_col, local_col_node) in enumerate(CartesianIndices(col_size))
        col_node = local_col_node + first_col_node - oneunit(first_col_node)
        neighboring_rows = cartesian_neighbor_nodes(local_col_node + col_offset, row_size, sparsity_radius)
        matrix_neighboring_rows = cartesian_neighbor_nodes(col_node, mesh_size, sparsity_radius)
        for b in 1:col_ndofs
            matrix_col_start = first(cartesian_matrix_column_slots(assembler, b, col_node))
            for (local_row, local_row_node) in enumerate(neighboring_rows)
                local_slot = node_colstarts[local_col] + ((local_row - 1) * col_ndofs + b - 1) * row_ndofs
                row_node = local_row_node + first_row_node - oneunit(first_row_node)
                matrix_slot = matrix_col_start + cartesian_slot_offset(row_node, matrix_neighboring_rows, row_slots_per_node)
                add_entry_values!(SerialScatter(), matrix_values, matrix_slot, values, local_slot, row_components, row_ndofs)
            end
        end
    end

    matrix
end

@noinline function check_block_matrix_scatter(assembler::CartesianSparseMatrixAssembler, buffer::BlockMatrixBuffer, row_nodes::CartesianIndices, col_nodes::CartesianIndices)
    (; row_dof_table, col_dof_table, sparsity_radius) = assembler
    (; key) = buffer
    (; row_size, col_size, col_offset, row_ndofs, col_ndofs) = key
    mesh_size = Base.tail(size(row_dof_table))
    size(row_nodes) == row_size || throw(DimensionMismatch("row support size does not match block matrix buffer"))
    size(col_nodes) == col_size || throw(DimensionMismatch("column support size does not match block matrix buffer"))
    first(col_nodes) - first(row_nodes) == col_offset || throw(DimensionMismatch("support offset does not match block matrix buffer"))
    Base.tail(size(col_dof_table)) == mesh_size || throw(DimensionMismatch("row and column DoF tables must have the same mesh size"))
    size(row_dof_table, 1) == row_ndofs || throw(DimensionMismatch("row DoF count does not match block matrix buffer"))
    size(col_dof_table, 1) == col_ndofs || throw(DimensionMismatch("column DoF count does not match block matrix buffer"))
    sparsity_radius == key.sparsity_radius || throw(DimensionMismatch("sparsity radius does not match block matrix buffer"))
    mesh_nodes = CartesianIndices(mesh_size)
    checkbounds(mesh_nodes, row_nodes)
    checkbounds(mesh_nodes, col_nodes)
    nothing
end

# -- final scatter --

function finish_assembly!(assembler::CartesianSparseMatrixAssembler, buffer::BlockMatrixBuffer, pool::BlockMatrixBufferPool, row_nodes, col_nodes)
    @_propagate_inbounds_meta
    try
        scatter!(assembler, buffer, row_nodes, col_nodes)
    finally
        release!(pool, buffer)
    end
end

finish_assembly!(assembler::GenericMatrixAssembler, ::Nothing, pool::BlockMatrixBufferPool, row_nodes, col_nodes) = assembler.matrix

