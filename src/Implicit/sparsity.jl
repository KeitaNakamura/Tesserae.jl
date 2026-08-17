# -----------------------------------------------------------------------------
#  Sparse matrices
# -----------------------------------------------------------------------------

# ---- construction ----

"""
    create_sparse_matrix(mesh; ndofs)
    create_sparse_matrix((rowmesh, colmesh); ndofs=(row_ndofs, col_ndofs))
    create_sparse_matrix(basis, mesh; ndofs)

Create a sparse matrix.
Since the created matrix accounts for all nodes in the mesh,
it needs to be extracted for active nodes using the `DofMap`.
`ndofs` specifies the number of DoFs for a field and must be provided explicitly.
For a mesh pair, the first mesh defines the rows and the second defines the
columns.

```jldoctest
julia> mesh = CartesianMesh(1, (0,10), (0,10));

julia> A = create_sparse_matrix(BSpline(Linear()), mesh; ndofs = 1)
121×121 SparseArrays.SparseMatrixCSC{Float64, Int64} with 961 stored entries:
⎡⠻⣦⡀⠘⢳⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎤
⎢⣀⠈⠻⣦⡀⠘⢳⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠙⢶⣀⠈⠻⣦⡀⠙⢷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠙⢶⣄⠈⠻⣦⡀⠙⠷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠙⢷⣄⠈⠻⣦⡀⠉⠷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠙⢧⡄⠈⠛⣤⡀⠉⠣⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠙⢧⡄⠈⠻⣦⡀⠉⠷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⢦⡄⠈⠱⣦⡀⠉⠷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢧⡄⠈⠻⣦⡀⠙⢷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢧⣄⠈⠻⣦⡀⠙⢷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢷⣄⠈⠻⣦⡀⠘⢳⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢷⣀⠈⠻⣦⡀⠘⢳⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢶⣀⠈⠻⢆⡀⠘⠳⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢶⣀⠈⠻⣦⡀⠘⢳⣄⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢢⣀⠈⠛⣤⡀⠘⢳⣄⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢶⣀⠈⠻⣦⡀⠙⢷⣄⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢶⣄⠈⠻⣦⡀⠙⠷⣄⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢷⣄⠈⠻⣦⡀⠉⠷⣄⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢧⡄⠈⠻⣦⡀⠉⎥
⎣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢧⡄⠈⠻⣦⎦

julia> dofmask = falses(1, size(mesh)...);

julia> dofmask[:,1:3,1:3] .= true;

julia> free = DofMap(dofmask);

julia> extract(A, free)
9×9 SparseArrays.SparseMatrixCSC{Float64, Int64} with 49 stored entries:
 0.0  0.0   ⋅   0.0  0.0   ⋅    ⋅    ⋅    ⋅
 0.0  0.0  0.0  0.0  0.0  0.0   ⋅    ⋅    ⋅
  ⋅   0.0  0.0   ⋅   0.0  0.0   ⋅    ⋅    ⋅
 0.0  0.0   ⋅   0.0  0.0   ⋅   0.0  0.0   ⋅
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
  ⋅   0.0  0.0   ⋅   0.0  0.0   ⋅   0.0  0.0
  ⋅    ⋅    ⋅   0.0  0.0   ⋅   0.0  0.0   ⋅
  ⋅    ⋅    ⋅   0.0  0.0  0.0  0.0  0.0  0.0
  ⋅    ⋅    ⋅    ⋅   0.0  0.0   ⋅   0.0  0.0
```
"""
function create_sparse_matrix end

# -- helpers --

function append_dofs!(I, J, row_dofs, col_dofs, row_offset, col_offset)
    for col_dof in col_dofs, row_dof in row_dofs
        push!(I, row_offset + row_dof)
        push!(J, col_offset + col_dof)
    end
    nothing
end

function _create_cell_support_sparse_matrix(::Type{T}, mesh, ndofs::Int) where {T}
    _create_cell_support_sparse_matrix(T, mesh, (ndofs, ndofs))
end

function _create_cell_support_sparse_matrix(::Type{T}, mesh, ndofs::Tuple{Int, Int}) where {T}
    I, J = Int[], Int[]
    _append_sparse_pattern!(I, J, 0, 0, mesh, ndofs[1], ndofs[2])
    sparse(I, J, zeros(T, length(I)), ndofs[1] * length(mesh), ndofs[2] * length(mesh))
end

function _append_sparse_pattern!(I, J, row_offset, col_offset, mesh::AbstractCellMesh, row_ndofs, col_ndofs)
    row_dofs = LinearIndices((row_ndofs, length(mesh)))
    col_dofs = LinearIndices((col_ndofs, length(mesh)))
    for cell in cells(mesh)
        cell_nodes = supportnodes(mesh, cell)
        append_dofs!(I, J, row_dofs[:, cell_nodes], col_dofs[:, cell_nodes], row_offset, col_offset)
    end
    nothing
end

function _append_sparse_pattern!(I, J, row_offset, col_offset, rowmesh::AbstractCellMesh, colmesh::AbstractCellMesh, row_ndofs, col_ndofs)
    throw(ArgumentError("all field meshes must use compatible discretizations"))
end

# -- MPM --

create_sparse_matrix(basis::Basis, mesh::CartesianMesh; ndofs) = _create_sparse_matrix(Float64, basis, mesh, ndofs)
create_sparse_matrix(::Type{T}, basis::Basis, mesh::CartesianMesh; ndofs) where {T} = _create_sparse_matrix(T, basis, mesh, ndofs)

function _create_sparse_matrix(::Type{T}, basis::Basis, mesh::CartesianMesh{dim}, ndofs::Int) where {T, dim}
    _create_sparse_matrix(T, basis, mesh, (ndofs, ndofs))
end

function _create_sparse_matrix(::Type{T}, basis::Basis, mesh::CartesianMesh{dim}, ndofs::Tuple{Int, Int}) where {T, dim}
    row_ndofs, col_ndofs = ndofs
    I, J = Int[], Int[]
    _append_sparse_pattern!(I, J, 0, 0, basis, mesh, row_ndofs, col_ndofs)
    sparse(I, J, zeros(T, length(I)), row_ndofs * length(mesh), col_ndofs * length(mesh))
end

function _append_sparse_pattern!(I, J, row_offset, col_offset, basis::Basis, mesh::CartesianMesh{N}, row_ndofs, col_ndofs) where {N}
    mesh_size = size(mesh)
    node_ids = LinearIndices(mesh_size)
    mesh_nodes = CartesianIndices(mesh_size)
    radius = (support_width(basis) - 1) * oneunit(CartesianIndex{N})
    for row_node in mesh_nodes
        row_dofs = node_dofs(node_ids[row_node], row_ndofs)
        for col_node in intersect((row_node - radius):(row_node + radius), mesh_nodes)
            col_dofs = node_dofs(node_ids[col_node], col_ndofs)
            append_dofs!(I, J, row_dofs, col_dofs, row_offset, col_offset)
        end
    end
    nothing
end

function node_dofs(node, ndofs)
    (ndofs * (node - 1) + 1):(ndofs * node)
end

# -- FEM and IGA --

function create_sparse_matrix(::Type{T}, (rowmesh,colmesh)::Tuple{Vararg{AbstractCellMesh, 2}}; ndofs::Tuple{Int, Int}) where {T}
    I, J = Int[], Int[]
    _append_sparse_pattern!(I, J, 0, 0, rowmesh, colmesh, ndofs[1], ndofs[2])
    sparse(I, J, zeros(T, length(I)), ndofs[1] * length(rowmesh), ndofs[2] * length(colmesh))
end

function create_sparse_matrix(meshes::Tuple{Vararg{AbstractCellMesh, 2}}; ndofs::Tuple{Int, Int})
    create_sparse_matrix(Float64, meshes; ndofs)
end

# -- FEM --

create_sparse_matrix(::Type{T}, mesh::FEMesh; ndofs) where {T} = _create_cell_support_sparse_matrix(T, mesh, ndofs)
create_sparse_matrix(mesh::FEMesh; ndofs) = create_sparse_matrix(Float64, mesh; ndofs)

function _append_sparse_pattern!(I, J, row_offset, col_offset, rowmesh::FEMesh, colmesh::FEMesh, row_ndofs, col_ndofs)
    rowmesh === colmesh && return _append_sparse_pattern!(I, J, row_offset, col_offset, rowmesh, row_ndofs, col_ndofs)
    _reference_cell_family(cellshape(rowmesh)) === _reference_cell_family(cellshape(colmesh)) || throw(ArgumentError("FEM meshes must use the same reference-cell family"))
    ncells(rowmesh) == ncells(colmesh) || throw(DimensionMismatch("FEM meshes must have the same number of cells"))

    row_dofs = LinearIndices((row_ndofs, length(rowmesh)))
    col_dofs = LinearIndices((col_ndofs, length(colmesh)))
    row_primary_nodes = primarynodes_indices(cellshape(rowmesh))
    col_primary_nodes = primarynodes_indices(cellshape(colmesh))
    for (row_cell, col_cell) in zip(cells(rowmesh), cells(colmesh))
        row_nodes = supportnodes(rowmesh, row_cell)
        col_nodes = supportnodes(colmesh, col_cell)
        rowmesh[row_nodes[row_primary_nodes]] ≈ colmesh[col_nodes[col_primary_nodes]] || throw(ArgumentError("FEM meshes must describe the same cells in the same order and orientation; cell $row_cell does not match"))
        append_dofs!(I, J, row_dofs[:, row_nodes], col_dofs[:, col_nodes], row_offset, col_offset)
    end
    nothing
end

# -- IGA --

create_sparse_matrix(::IGABasis, mesh::IGAMesh{dim}; ndofs) where {dim} = _create_sparse_matrix(Float64, mesh, ndofs)
create_sparse_matrix(::Type{T}, ::IGABasis, mesh::IGAMesh{dim}; ndofs) where {T, dim} = _create_sparse_matrix(T, mesh, ndofs)

_create_sparse_matrix(::Type{T}, ::IGABasis, mesh::IGAMesh, ndofs::Int) where {T} = _create_sparse_matrix(T, mesh, ndofs)
_create_sparse_matrix(::Type{T}, ::IGABasis, mesh::IGAMesh, ndofs::Tuple{Int, Int}) where {T} = _create_sparse_matrix(T, mesh, ndofs)
_create_sparse_matrix(::Type{T}, mesh::IGAMesh, ndofs::Int) where {T} = _create_cell_support_sparse_matrix(T, mesh, ndofs)
_create_sparse_matrix(::Type{T}, mesh::IGAMesh, ndofs::Tuple{Int, Int}) where {T} = _create_cell_support_sparse_matrix(T, mesh, ndofs)
create_sparse_matrix(::Type{T}, mesh::IGAMesh{dim}; ndofs) where {T, dim} = _create_sparse_matrix(T, mesh, ndofs)
create_sparse_matrix(mesh::IGAMesh{dim}; ndofs) where {dim} = create_sparse_matrix(Float64, mesh; ndofs)

function _append_sparse_pattern!(I, J, row_offset, col_offset, rowmesh::IGAMesh{dim, pdim}, colmesh::IGAMesh{dim, pdim}, row_ndofs, col_ndofs) where {dim, pdim}
    rowmesh === colmesh && return _append_sparse_pattern!(I, J, row_offset, col_offset, rowmesh, row_ndofs, col_ndofs)
    check_matching_cell_partitions(rowmesh, colmesh)

    row_dofs = LinearIndices((row_ndofs, length(rowmesh)))
    col_dofs = LinearIndices((col_ndofs, length(colmesh)))
    for (row_cell, col_cell) in zip(cells(rowmesh), cells(colmesh))
        row_nodes = supportnodes(rowmesh, row_cell)
        col_nodes = supportnodes(colmesh, col_cell)
        append_dofs!(I, J, row_dofs[:, row_nodes], col_dofs[:, col_nodes], row_offset, col_offset)
    end
    nothing
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

# ---- extraction ----

"""
    extract(matrix, dofmap_row, dofmap_col = dofmap_row)

Extract the active degrees of freedom of a matrix.
"""
function extract(matrix::AbstractMatrix, dofmap_i, dofmap_j = dofmap_i)
    I, J = _indices_for_extract(matrix, dofmap_i, dofmap_j)
    row_parent_indices, col_parent_indices = matrix_parent_indices(matrix)
    matrix_parent(matrix)[mapped_index(row_parent_indices, I), mapped_index(col_parent_indices, J)]
end
function extract(::typeof(view), matrix::AbstractMatrix, dofmap_i, dofmap_j = dofmap_i)
    I, J = _indices_for_extract(matrix, dofmap_i, dofmap_j)
    row_parent_indices, col_parent_indices = matrix_parent_indices(matrix)
    view(matrix_parent(matrix), mapped_index(row_parent_indices, I), mapped_index(col_parent_indices, J))
end

function _indices_for_extract(matrix::AbstractMatrix, dofmap_i::Union{AbstractDofMap, Colon}, dofmap_j::Union{AbstractDofMap, Colon})
    check_dofmap_size(size(matrix, 1), dofmap_i)
    check_dofmap_size(size(matrix, 2), dofmap_j)
    dofs(dofmap_i), dofs(dofmap_j)
end

function _indices_for_extract(blocks::SparseMatrixBlocks, dofmap_i::Union{AbstractDofMap, Colon}, dofmap_j::Union{AbstractDofMap, Colon})
    check_block_dofmap(blocks, dofmap_i)
    check_block_dofmap(blocks, dofmap_j)
    dofs(dofmap_i), dofs(dofmap_j)
end

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
    @assert issorted(I)
    @assert size(K) == map(length, (I, J))
    @inbounds @views A[I,J] .+= K
end

