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

function _create_sparse_matrix(::Type{T}, basis::Basis, mesh::CartesianMesh, ndofs::Int) where {T}
    _create_sparse_matrix(T, basis, mesh, (ndofs, ndofs))
end

function _create_sparse_matrix(::Type{T}, basis::Basis, mesh::CartesianMesh, ndofs::Tuple{Int, Int}) where {T}
    row_ndofs, col_ndofs = ndofs
    I, J = Int[], Int[]
    _append_sparse_pattern!(I, J, 0, 0, basis, mesh, row_ndofs, col_ndofs)
    sparse(I, J, zeros(T, length(I)), row_ndofs * length(mesh), col_ndofs * length(mesh))
end

function _append_sparse_pattern!(I, J, row_offset, col_offset, basis::Basis, mesh::CartesianMesh{N}, row_ndofs, col_ndofs) where {N}
    mesh_size = size(mesh)
    node_ids = LinearIndices(mesh_size)
    mesh_nodes = CartesianIndices(mesh_size)
    for row_node in mesh_nodes
        row_dofs = node_dofs(node_ids[row_node], row_ndofs)
        for col_node in cartesian_neighbor_nodes(row_node, mesh_size, support_width(basis) - 1)
            col_dofs = node_dofs(node_ids[col_node], col_ndofs)
            append_dofs!(I, J, row_dofs, col_dofs, row_offset, col_offset)
        end
    end
    nothing
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

create_sparse_matrix(::IGABasis, mesh::IGAMesh; ndofs) = _create_sparse_matrix(Float64, mesh, ndofs)
create_sparse_matrix(::Type{T}, ::IGABasis, mesh::IGAMesh; ndofs) where {T} = _create_sparse_matrix(T, mesh, ndofs)

_create_sparse_matrix(::Type{T}, mesh::IGAMesh, ndofs::Int) where {T} = _create_cell_support_sparse_matrix(T, mesh, ndofs)
_create_sparse_matrix(::Type{T}, mesh::IGAMesh, ndofs::Tuple{Int, Int}) where {T} = _create_cell_support_sparse_matrix(T, mesh, ndofs)
create_sparse_matrix(::Type{T}, mesh::IGAMesh; ndofs) where {T} = _create_sparse_matrix(T, mesh, ndofs)
create_sparse_matrix(mesh::IGAMesh; ndofs) = create_sparse_matrix(Float64, mesh; ndofs)

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
