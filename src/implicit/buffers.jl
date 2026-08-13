# -----------------------------------------------------------------------------
#  LocalMatrixBuffer
# -----------------------------------------------------------------------------

# ---- construction ----

struct LocalMatrixBuffer{M <: Matrix}
    matrix::M
    row_ndofs::Int
    col_ndofs::Int
end

function local_matrix_cache(matrix, dof_table_i, weights_i, dof_table_j, weights_j)
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
    @boundscheck size(matrix) == dims || throw(DimensionMismatch("local matrix size changed during assembly"))
    LocalMatrixBuffer(matrix, row_ndofs, col_ndofs)
end

# ---- DoF indexing ----

matrix_row_dofs(buffer::LocalMatrixBuffer, ip) = (@_propagate_inbounds_meta; local_dofs(buffer.row_ndofs, ip))
matrix_col_dofs(buffer::LocalMatrixBuffer, jp) = (@_propagate_inbounds_meta; local_dofs(buffer.col_ndofs, jp))
matrix_row_dofs(::Nothing, ip) = nothing
matrix_col_dofs(::Nothing, jp) = nothing

function local_dofs(ndofs::Int, index::Integer)
    @_propagate_inbounds_meta
    vec(view(LinearIndices((ndofs, index)), :, index))
end

# ---- assembly ----

function assemble_first!(assembler, buffer::LocalMatrixBuffer, orientation, row_node, col_node, I, J, value)
    @_propagate_inbounds_meta
    buffer.matrix[I,J] .= value
    buffer
end

function assemble_add!(assembler, buffer::LocalMatrixBuffer, orientation, row_node, col_node, I, J, value)
    @_propagate_inbounds_meta
    @views buffer.matrix[I,J] .+= value
    buffer
end

function finish_assembly!(assembler, buffer::LocalMatrixBuffer, orientation, row_nodes, col_nodes)
    @_propagate_inbounds_meta
    row_nodes, col_nodes = orientation((row_nodes, col_nodes))
    add!(assembler, row_nodes, col_nodes, orient_matrix_entry(orientation, buffer.matrix))
end

function assemble_first!(assembler, ::Nothing, orientation, row_node, col_node, I, J, value)
    @_propagate_inbounds_meta
    row_node, col_node = orientation((row_node, col_node))
    add_entry!(assembler, row_node, col_node, orient_matrix_entry(orientation, value))
end

function assemble_add!(assembler, buffer::Nothing, orientation, row_node, col_node, I, J, value)
    @_propagate_inbounds_meta
    assemble_first!(assembler, buffer, orientation, row_node, col_node, I, J, value)
end

finish_assembly!(assembler, ::Nothing, orientation, row_nodes, col_nodes) = assembler.matrix

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

# Canonical Cartesian CSC matrices, component views, and block views use a
# block buffer. Other matrices are written directly within the thread-safe
# block schedule.
function block_matrix_buffer(assembler::CartesianSparseMatrixAssembler, assembly::BlockAssembly, orientation)
    row_nodes, col_nodes = orientation((assembly.nodes_i, assembly.nodes_j))
    acquire!(assembly.matrix_buffer_pool, BlockMatrixBufferKey(assembler, row_nodes, col_nodes))
end

block_matrix_buffer(::GenericMatrixAssembler, ::BlockAssembly, orientation) = nothing

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
    add_entry_values!(values, slot, value)

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

    (; matrix, row_dof_table, storage_row_ndofs, sparsity_radius) = assembler
    storage = matrix_storage(matrix)
    row_storage_indices = cartesian_matrix_row_storage_indices(assembler)
    (; values, node_colstarts, key) = buffer
    (; row_size, col_size, col_offset, row_ndofs, col_ndofs) = key

    mesh_size = Base.tail(size(row_dof_table))
    matrix_values = nonzeros(storage)
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
                matrix_slot = matrix_col_start + cartesian_slot_offset(row_node, matrix_neighboring_rows, storage_row_ndofs)
                add_entry_values!(matrix_values, matrix_slot, values, local_slot, row_storage_indices, row_ndofs)
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

# -- route dispatch --

# Canonical Cartesian entries are accumulated in the block buffer.
function assemble_block_entry!(assembler::CartesianSparseMatrixAssembler, buffer::BlockMatrixBuffer, assembly::BlockAssembly, orientation, row_node, col_node, value)
    @_propagate_inbounds_meta
    row_nodes, col_nodes = orientation((assembly.nodes_i, assembly.nodes_j))
    row_node, col_node = orientation((row_node, col_node))
    add_entry!(buffer, row_nodes, col_nodes, row_node, col_node, orient_matrix_entry(orientation, value))
end

# Generic matrices are written directly within the thread-safe block schedule.
function assemble_block_entry!(assembler::GenericMatrixAssembler, ::Nothing, assembly::BlockAssembly, orientation, row_node, col_node, value)
    @_propagate_inbounds_meta
    row_node, col_node = orientation((row_node, col_node))
    add_entry!(assembler, row_node, col_node, orient_matrix_entry(orientation, value))
end

function finish_block_assembly!(assembler::CartesianSparseMatrixAssembler, buffer::BlockMatrixBuffer, assembly::BlockAssembly, orientation)
    @_propagate_inbounds_meta
    row_nodes, col_nodes = orientation((assembly.nodes_i, assembly.nodes_j))
    try
        scatter!(assembler, buffer, row_nodes, col_nodes)
    finally
        release!(assembly.matrix_buffer_pool, buffer)
    end
end

function finish_block_assembly!(assembler::GenericMatrixAssembler, ::Nothing, assembly::BlockAssembly, orientation)
    assembler.matrix
end

