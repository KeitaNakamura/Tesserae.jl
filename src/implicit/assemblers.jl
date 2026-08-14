# -----------------------------------------------------------------------------
#  Matrix assemblers
# -----------------------------------------------------------------------------

# ---- matrix entries ----

orient_matrix_entry(::typeof(identity), value) = value
orient_matrix_entry(::typeof(reverse), value) = value'

@noinline function check_matrix_entry_size(::Number, row_ndofs, col_ndofs)
    (row_ndofs, col_ndofs) == (1, 1) || throw(DimensionMismatch("scalar value requires a scalar matrix entry"))
end
@noinline function check_matrix_entry_size(value::AbstractVector, row_ndofs, col_ndofs)
    expected_length = col_ndofs == 1 ? row_ndofs : col_ndofs
    (row_ndofs == 1 || col_ndofs == 1) && length(value) == expected_length ||
        throw(DimensionMismatch("vector value is incompatible with matrix entry dimensions"))
end
@noinline function check_matrix_entry_size(value::AbstractMatrix, row_ndofs, col_ndofs)
    size(value) == (row_ndofs, col_ndofs) || throw(DimensionMismatch("matrix value is incompatible with matrix entry dimensions"))
end

function add_entry_values!(destination, destination_slot, source, source_slot, row_components, count)
    @_propagate_inbounds_meta
    for index in 1:count
        destination[destination_slot + mapped_index(row_components, index) - 1] += source[source_slot]
        source_slot += 1
    end
    nothing
end
function add_entry_values!(destination, destination_slot, source)
    @_propagate_inbounds_meta
    add_entry_values!(destination, destination_slot, source, firstindex(source), Base.OneTo(length(source)), length(source))
end

# ---- CartesianSparseMatrixAssembler ----

# -- destination layout --

# The DoF tables index the logical destination. `row_slots_per_node` is the
# stride between nodes in the destination's nonzeros run. It is the parent's row
# DoFs per node for a plain matrix and for a CSC view, but the block's own for a
# `SparseMatrixBlockView`, whose `column_slots` already restrict each column to
# the block -- so it is deliberately not named after the parent.
struct CartesianSparseMatrixAssembler{M <: AbstractMatrix, D <: LinearIndices}
    matrix::M
    row_dof_table::D
    col_dof_table::D
    row_slots_per_node::Int
    sparsity_radius::Int
end

function cartesian_matrix_column_slots(assembler::CartesianSparseMatrixAssembler, b, col_node)
    @_propagate_inbounds_meta
    (; matrix, col_dof_table) = assembler
    parent_matrix = matrix_parent(matrix)
    col_parent_indices = last(matrix_parent_indices(matrix))
    col = mapped_index(col_parent_indices, col_dof_table[b,col_node])
    nzrange(parent_matrix, col)
end

function cartesian_matrix_column_slots(assembler::CartesianSparseMatrixAssembler{<:SparseMatrixBlockView}, b, col_node)
    @_propagate_inbounds_meta
    (; matrix, col_dof_table) = assembler
    matrix.column_slots[col_dof_table[b,col_node]]
end

# Maps a within-node row DoF (`1:row_ndofs`) to the parent's row DoF component.
# Not the parent's index list: only the first `row_ndofs` entries are ever read,
# and they are the components at every node because
# `check_cartesian_sparse_matrix_view_indices` requires the view to select the
# same ones node by node. Consumed only by `add_entry_values!`.
@inline function cartesian_matrix_row_components(assembler::CartesianSparseMatrixAssembler)
    first(matrix_parent_indices(assembler.matrix))
end

@inline function cartesian_matrix_row_components(assembler::CartesianSparseMatrixAssembler{<:SparseMatrixBlockView})
    Base.OneTo(size(assembler.row_dof_table, 1))
end

function cartesian_matrix_parent_row(assembler::CartesianSparseMatrixAssembler, a, row_node)
    @_propagate_inbounds_meta
    (; matrix, row_dof_table) = assembler
    mapped_index(first(matrix_parent_indices(matrix)), row_dof_table[a,row_node])
end

# -- construction --

function CartesianSparseMatrixAssembler(A::SparseMatrixCSC, mesh_size::Dims, sparsity_radius::Int)
    node_count = prod(mesh_size)
    node_count > 0 || throw(ArgumentError("mesh must contain at least one node"))
    row_dofs_per_node, row_remainder = divrem(size(A, 1), node_count)
    col_dofs_per_node, col_remainder = divrem(size(A, 2), node_count)
    iszero(row_remainder) && iszero(col_remainder) || throw(DimensionMismatch("matrix dimensions must be multiples of the number of mesh nodes"))
    row_dofs_per_node > 0 && col_dofs_per_node > 0 || throw(DimensionMismatch("matrix must have at least one row and column DoF per node"))
    sparsity_radius ≥ 0 || throw(ArgumentError("sparsity radius must be nonnegative"))
    row_dof_table = LinearIndices((row_dofs_per_node, mesh_size...))
    col_dof_table = LinearIndices((col_dofs_per_node, mesh_size...))
    assembler = CartesianSparseMatrixAssembler(
        A,
        row_dof_table,
        col_dof_table,
        row_dofs_per_node,
        sparsity_radius,
    )
    has_cartesian_sparse_pattern(assembler) || throw(ArgumentError("Cartesian sparse matrix must use the canonical sparsity pattern"))
    assembler
end

function cartesian_sparsity_radius(row_mesh, col_mesh, row_basis, col_basis)
    size(row_mesh) == size(col_mesh) || throw(DimensionMismatch("row and column meshes must have the same size"))
    sparsity_radius = support_width(row_basis) - 1
    sparsity_radius == support_width(col_basis) - 1 || throw(ArgumentError("row and column bases must have the same support width"))
    sparsity_radius
end

function CartesianSparseMatrixAssembler(A::SparseMatrixCSC, row_mesh::CartesianMesh{N}, col_mesh::CartesianMesh{N}, row_basis::Basis, col_basis::Basis) where {N}
    sparsity_radius = cartesian_sparsity_radius(row_mesh, col_mesh, row_basis, col_basis)
    CartesianSparseMatrixAssembler(A, size(row_mesh), sparsity_radius)
end

# -- sparsity pattern --

function cartesian_neighbor_nodes(node::CartesianIndex{N}, mesh_size::Dims{N}, sparsity_radius::Int) where {N}
    shift = sparsity_radius * oneunit(CartesianIndex{N})
    ((node-shift) : (node+shift)) ∩ CartesianIndices(mesh_size)
end

function cartesian_slot_offset(node, neighboring_nodes, slots_per_node)
    # Assembly checks establish that `node` belongs to `neighboring_nodes`.
    local_node = node - first(neighboring_nodes) + oneunit(node)
    @inbounds (LinearIndices(neighboring_nodes)[local_node] - 1) * slots_per_node
end

function has_cartesian_sparse_pattern(assembler::CartesianSparseMatrixAssembler)
    (; matrix, row_dof_table, col_dof_table, sparsity_radius) = assembler
    parent_matrix = matrix_parent(matrix)
    mesh_size = Base.tail(size(row_dof_table))
    row_ndofs = size(row_dof_table, 1)
    col_ndofs = size(col_dof_table, 1)
    rows = rowvals(parent_matrix)
    for col_node in CartesianIndices(mesh_size), b in 1:col_ndofs
        slots = cartesian_matrix_column_slots(assembler, b, col_node)
        k = first(slots)
        stop = last(slots) + 1
        for row_node in cartesian_neighbor_nodes(col_node, mesh_size, sparsity_radius), a in 1:row_ndofs
            k < stop || return false
            rows[k] == cartesian_matrix_parent_row(assembler, a, row_node) || return false
            k += 1
        end
        k == stop || return false
    end
    true
end

function has_cartesian_sparse_pattern(assembler::CartesianSparseMatrixAssembler{<:SparseMatrixCSCView})
    (; matrix, row_dof_table, row_slots_per_node, sparsity_radius) = assembler
    mesh_size = Base.tail(size(row_dof_table))
    parent_col_ndofs = size(parent(matrix), 2) ÷ prod(mesh_size)
    parent_row_dof_table = LinearIndices((row_slots_per_node, mesh_size...))
    parent_col_dof_table = LinearIndices((parent_col_ndofs, mesh_size...))
    parent_assembler = CartesianSparseMatrixAssembler(
        parent(matrix),
        parent_row_dof_table,
        parent_col_dof_table,
        row_slots_per_node,
        sparsity_radius,
    )
    has_cartesian_sparse_pattern(parent_assembler)
end

# -- assembly --

# The canonical Cartesian CSC pattern allows its destination slots to be computed directly.
@inline function add_entry!(assembler::CartesianSparseMatrixAssembler, row_node::CartesianIndex, col_node::CartesianIndex, value)
    @boundscheck check_cartesian_matrix_entry(assembler, row_node, col_node, value)

    (; matrix, row_dof_table, col_dof_table, row_slots_per_node, sparsity_radius) = assembler
    parent_matrix = matrix_parent(matrix)
    row_components = cartesian_matrix_row_components(assembler)
    mesh_size = Base.tail(size(row_dof_table))
    row_ndofs = size(row_dof_table, 1)
    col_ndofs = size(col_dof_table, 1)
    neighboring_nodes = cartesian_neighbor_nodes(col_node, mesh_size, sparsity_radius)

    row_offset = cartesian_slot_offset(row_node, neighboring_nodes, row_slots_per_node)
    values = nonzeros(parent_matrix)

    @inbounds for b in 1:col_ndofs
        slot = first(cartesian_matrix_column_slots(assembler, b, col_node)) + row_offset
        add_entry_values!(values, slot, value, (b - 1) * row_ndofs + 1, row_components, row_ndofs)
    end

    matrix
end

@noinline function check_cartesian_matrix_entry(assembler::CartesianSparseMatrixAssembler, row_node::CartesianIndex, col_node::CartesianIndex, value)
    (; row_dof_table, col_dof_table, sparsity_radius) = assembler
    mesh_size = Base.tail(size(row_dof_table))
    mesh_nodes = CartesianIndices(mesh_size)
    checkbounds(mesh_nodes, row_node)
    checkbounds(mesh_nodes, col_node)
    row_node ∈ cartesian_neighbor_nodes(col_node, mesh_size, sparsity_radius) || throw(ArgumentError("matrix entry is outside the Cartesian sparsity pattern"))
    check_matrix_entry_size(value, size(row_dof_table, 1), size(col_dof_table, 1))
    nothing
end

# -- matrix views --

function CartesianSparseMatrixAssembler(matrix::SparseMatrixCSCView, row_mesh::CartesianMesh{N}, col_mesh::CartesianMesh{N}, row_basis::Basis, col_basis::Basis) where {N}
    parent_assembler = CartesianSparseMatrixAssembler(parent(matrix), row_mesh, col_mesh, row_basis, col_basis)
    row_dof_table, col_dof_table = matrix_dof_tables(matrix, row_mesh, col_mesh)
    assembler = CartesianSparseMatrixAssembler(
        matrix,
        row_dof_table,
        col_dof_table,
        parent_assembler.row_slots_per_node,
        parent_assembler.sparsity_radius,
    )
    check_cartesian_sparse_matrix_view(assembler)
    assembler
end

function CartesianSparseMatrixAssembler(matrix::SparseMatrixBlockView{T, Ti, P}, row_mesh::CartesianMesh{N}, col_mesh::CartesianMesh{N}, row_basis::Basis, col_basis::Basis) where {T, Ti, N, P <: CartesianSparseMatrixPattern}
    sparsity_radius = cartesian_sparsity_radius(row_mesh, col_mesh, row_basis, col_basis)
    matrix.pattern.mesh_size == size(row_mesh) || throw(DimensionMismatch("matrix block and mesh sizes must match"))
    matrix.pattern.sparsity_radius == sparsity_radius || throw(ArgumentError("matrix block and basis must use the same support width"))
    row_dof_table, col_dof_table = matrix_dof_tables(matrix, row_mesh, col_mesh)
    CartesianSparseMatrixAssembler(
        matrix,
        row_dof_table,
        col_dof_table,
        size(row_dof_table, 1),
        sparsity_radius,
    )
end

function CartesianSparseMatrixAssembler(::SparseMatrixBlockView, ::CartesianMesh, ::CartesianMesh, ::Basis, ::Basis)
    throw(ArgumentError("Cartesian assembly requires blocks created from a Cartesian mesh"))
end

@noinline function check_cartesian_sparse_matrix_view(assembler::CartesianSparseMatrixAssembler)
    (; matrix, row_dof_table, col_dof_table, row_slots_per_node) = assembler
    node_count = prod(Base.tail(size(row_dof_table)))
    parent_col_ndofs = size(parent(matrix), 2) ÷ node_count
    row_parent_indices, col_parent_indices = matrix_parent_indices(matrix)
    check_cartesian_sparse_matrix_view_indices(row_parent_indices, row_dof_table, row_slots_per_node)
    check_cartesian_sparse_matrix_view_indices(col_parent_indices, col_dof_table, parent_col_ndofs)
    nothing
end

@noinline function check_cartesian_sparse_matrix_view_indices(parent_indices, dof_table, parent_ndofs)
    ndofs = size(dof_table, 1)
    mesh_nodes = CartesianIndices(Base.tail(size(dof_table)))
    parent_dof_table = LinearIndices((parent_ndofs, size(mesh_nodes)...))
    first_node = first(mesh_nodes)
    for a in 1:ndofs
        component = parent_indices[dof_table[a,first_node]]
        component ∈ 1:parent_ndofs || throw(ArgumentError("matrix view must select the same DoF components at every node"))
        all(parent_indices[dof_table[b,first_node]] != component for b in 1:a-1) || throw(ArgumentError("matrix view must not select a DoF component more than once"))
        all(parent_indices[dof_table[a,node]] == parent_dof_table[component,node] for node in mesh_nodes) || throw(ArgumentError("matrix view must select the same DoF components at every node"))
    end
    nothing
end

# ---- GenericMatrixAssembler ----

struct GenericMatrixAssembler{M <: AbstractMatrix, D <: LinearIndices}
    matrix::M
    row_dof_table::D
    col_dof_table::D
end

function add!(assembler::GenericMatrixAssembler, row_nodes, col_nodes, local_matrix)
    @_propagate_inbounds_meta
    (; matrix, row_dof_table, col_dof_table) = assembler
    parent_matrix = matrix_parent(matrix)
    row_parent_indices, col_parent_indices = matrix_parent_indices(matrix)
    row_dofs, col_dofs = support_dofs(row_dof_table, row_nodes, col_dof_table, col_nodes)
    if row_dofs === col_dofs && row_parent_indices === col_parent_indices
        dofs = mapped_index(row_parent_indices, row_dofs)
        add!(parent_matrix, dofs, dofs, local_matrix)
    else
        add!(parent_matrix, mapped_index(row_parent_indices, row_dofs), mapped_index(col_parent_indices, col_dofs), local_matrix)
    end
    matrix
end

# GenericMatrixAssembler writes entries through ordinary `setindex!` on the parent.
@inline function add_entry!(assembler::GenericMatrixAssembler, row_node, col_node, value)
    (; matrix, row_dof_table, col_dof_table) = assembler
    parent_matrix = matrix_parent(matrix)
    row_parent_indices, col_parent_indices = matrix_parent_indices(matrix)
    row_ndofs = size(row_dof_table, 1)
    col_ndofs = size(col_dof_table, 1)
    @boundscheck check_matrix_entry_size(value, row_ndofs, col_ndofs)
    @inbounds for b in 1:col_ndofs, a in 1:row_ndofs
        row = mapped_index(row_parent_indices, row_dof_table[a,row_node])
        col = mapped_index(col_parent_indices, col_dof_table[b,col_node])
        parent_matrix[row,col] += value[(b - 1) * row_ndofs + a]
    end
    matrix
end

function support_dofs(table_i, nodes_i, table_j, nodes_j)
    @_propagate_inbounds_meta
    if size(table_i, 1) == size(table_j, 1) && nodes_i === nodes_j
        dofs = vec(table_i[:, nodes_i])
        return dofs, dofs
    else
        return vec(table_i[:, nodes_i]), vec(table_j[:, nodes_j])
    end
end

# ---- assembler construction ----

function matrix_assembler(matrix, row_mesh, col_mesh, row_basis, col_basis)
    row_dof_table, col_dof_table = matrix_dof_tables(matrix, row_mesh, col_mesh)
    GenericMatrixAssembler(matrix, row_dof_table, col_dof_table)
end
function matrix_assembler(matrix::Union{SparseMatrixCSC, SparseMatrixCSCView, SparseMatrixBlockView}, row_mesh::CartesianMesh, col_mesh::CartesianMesh, row_basis::Basis, col_basis::Basis)
    CartesianSparseMatrixAssembler(matrix, row_mesh, col_mesh, row_basis, col_basis)
end
function matrix_assembler(::SparseMatrixBlocks, row_mesh, col_mesh, row_basis, col_basis)
    throw(ArgumentError("@P2G_Matrix requires an individual matrix block; pass blocks[row, col] instead of blocks"))
end

function matrix_dof_tables(gmat, row_grid, col_grid)
    row_table = LinearIndices((size(gmat, 1) ÷ length(row_grid), size(row_grid)...))
    col_table = LinearIndices((size(gmat, 2) ÷ length(col_grid), size(col_grid)...))
    @assert size(gmat) == (length(row_table), length(col_table))
    row_table, col_table
end

