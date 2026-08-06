"""
    DofMap(mask::AbstractArray{Bool})

Create a degree of freedom (DoF) map from a `mask` of size `(ndofs, size(grid)...)`.
`ndofs` represents the number of DoFs for a field.

```jldoctest
julia> mesh = CartesianMesh(1, (0,2), (0,1));

julia> grid = generate_grid(@NamedTuple{x::Vec{2,Float64}, v::Vec{2,Float64}}, mesh);

julia> grid.v .= reshape(reinterpret(Vec{2,Float64}, 1.0:12.0), 3, 2)
3×2 Matrix{Vec{2, Float64}}:
 [1.0, 2.0]  [7.0, 8.0]
 [3.0, 4.0]  [9.0, 10.0]
 [5.0, 6.0]  [11.0, 12.0]

julia> dofmask = falses(2, size(grid)...);

julia> dofmask[1,1:2,:] .= true; # activate nodes

julia> dofmask[:,3,2] .= true; # activate nodes

julia> reinterpret(reshape, Vec{2,Bool}, dofmask)
3×2 reinterpret(reshape, Vec{2, Bool}, ::BitArray{3}) with eltype Vec{2, Bool}:
 [1, 0]  [1, 0]
 [1, 0]  [1, 0]
 [0, 0]  [1, 1]

julia> dofmap = DofMap(dofmask);

julia> dofmap(grid.v)
6-element view(reinterpret(reshape, Float64, ::Matrix{Vec{2, Float64}}), CartesianIndex{3}[CartesianIndex(1, 1, 1), CartesianIndex(1, 2, 1), CartesianIndex(1, 1, 2), CartesianIndex(1, 2, 2), CartesianIndex(1, 3, 2), CartesianIndex(2, 3, 2)]) with eltype Float64:
  1.0
  3.0
  7.0
  9.0
 11.0
 12.0
```
"""
struct DofMap{N, I <: AbstractVector{<: CartesianIndex}, J <: AbstractVector{<: CartesianIndex}}
    masksize::Dims{N}
    indices::I # (dof, x, y, z)
    indices4scalar::J # (dof, x, y, z)
end

function DofMap(mask::AbstractArray{Bool})
    masksize = size(mask)
    I = findall(mask)
    J = map(i -> CartesianIndex(1, Base.tail(Tuple(i))...), I)
    DofMap(masksize, I, J)
end
ndofs(dofmap::DofMap) = length(dofmap.indices)

function (dofmap::DofMap)(A::AbstractArray{T}) where {T <: Vec{1}}
    A′ = reshape(reinterpret(eltype(T), A), 1, size(A)...)
    @boundscheck checkbounds(A′, dofmap.indices)
    @inbounds view(A′, dofmap.indices)
end
function (dofmap::DofMap)(A::AbstractArray{T}) where {T <: Vec}
    A′ = reinterpret(reshape, eltype(T), A)
    @boundscheck checkbounds(A′, dofmap.indices)
    @inbounds view(A′, dofmap.indices)
end

function (dofmap::DofMap)(A::AbstractArray{T}) where {T <: Real}
    A′ = reshape(A, 1, size(A)...)
    @boundscheck checkbounds(A′, dofmap.indices4scalar)
    @inbounds view(A′, dofmap.indices4scalar)
end

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

julia> dofmap = DofMap(dofmask);

julia> extract(A, dofmap)
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
function create_sparse_matrix(basis::Basis, mesh::AbstractMesh; ndofs)
    _create_sparse_matrix(Float64, basis, mesh, ndofs)
end

function create_sparse_matrix(::Type{T}, basis::Basis, mesh::CartesianMesh; ndofs) where {T}
    _create_sparse_matrix(T, basis, mesh, ndofs)
end

function _create_sparse_matrix(::Type{T}, basis::Basis, mesh::CartesianMesh{dim}, ndofs::Int) where {T, dim}
    _create_sparse_matrix(T, basis, mesh, (ndofs, ndofs))
end

function _create_sparse_matrix(::Type{T}, basis::Basis, mesh::CartesianMesh{dim}, ndofs::Tuple{Int,Int}) where {T, dim}
    row_ndofs, col_ndofs = ndofs

    dims = size(mesh)
    nrows = row_ndofs * prod(dims)
    ncols = col_ndofs * prod(dims)

    I, J = Int[], Int[]
    LI, CI = LinearIndices(dims), CartesianIndices(dims)

    function gendofs(node_id, ndofs)
        first = ndofs * (node_id - 1) + 1
        last  = ndofs * node_id
        first:last
    end

    for i in CI
        unit = (support_width(basis) - 1) * oneunit(i)
        indices = intersect((i-unit):(i+unit), CI)
        idofs = gendofs(LI[i], row_ndofs)
        for j in indices
            jdofs = gendofs(LI[j], col_ndofs)
            append_dofs!(I, J, idofs, jdofs)
        end
    end

    sparse(I, J, zeros(T, length(I)), nrows, ncols)
end

function append_dofs!(I, J, idofs, jdofs)
    for jdof in jdofs
        append!(I, idofs)
        for _ in idofs
            push!(J, jdof)
        end
    end
end

function _create_cell_support_sparse_matrix(::Type{T}, mesh, ndofs::Int) where {T}
    _create_cell_support_sparse_matrix(T, mesh, (ndofs, ndofs))
end

function _create_cell_support_sparse_matrix(::Type{T}, mesh, ndofs::Tuple{Int,Int}) where {T}
    gdofs1 = LinearIndices((ndofs[1], length(mesh)))
    gdofs2 = LinearIndices((ndofs[2], length(mesh)))

    I, J = Int[], Int[]
    for cell in cells(mesh)
        cellnodes = supportnodes(mesh, cell)
        append_dofs!(I, J, gdofs1[:, cellnodes], gdofs2[:, cellnodes])
    end

    sparse(I, J, zeros(T, length(I)), length(gdofs1), length(gdofs2))
end

function create_sparse_matrix(::IGABasis, mesh::IGAMesh{dim}; ndofs) where {dim}
    _create_sparse_matrix(Float64, mesh, ndofs)
end
function create_sparse_matrix(::Type{T}, ::IGABasis, mesh::IGAMesh{dim}; ndofs) where {T, dim}
    _create_sparse_matrix(T, mesh, ndofs)
end
_create_sparse_matrix(::Type{T}, ::IGABasis, mesh::IGAMesh, ndofs::Int) where {T} = _create_sparse_matrix(T, mesh, ndofs)
_create_sparse_matrix(::Type{T}, ::IGABasis, mesh::IGAMesh, ndofs::Tuple{Int,Int}) where {T} = _create_sparse_matrix(T, mesh, ndofs)
_create_sparse_matrix(::Type{T}, mesh::IGAMesh, ndofs::Int) where {T} = _create_cell_support_sparse_matrix(T, mesh, ndofs)
_create_sparse_matrix(::Type{T}, mesh::IGAMesh, ndofs::Tuple{Int,Int}) where {T} = _create_cell_support_sparse_matrix(T, mesh, ndofs)
create_sparse_matrix(::Type{T}, mesh::IGAMesh{dim}; ndofs) where {T, dim} = _create_sparse_matrix(T, mesh, ndofs)
create_sparse_matrix(mesh::IGAMesh{dim}; ndofs) where {dim} = create_sparse_matrix(Float64, mesh; ndofs)

function create_sparse_matrix(::Type{T}, (rowmesh,colmesh)::Tuple{IGAMesh{dim,pdim}, IGAMesh{dim,pdim}}; ndofs::Tuple{Int, Int}) where {T, dim, pdim}
    rowmesh === colmesh && return _create_cell_support_sparse_matrix(T, rowmesh, ndofs)
    check_matching_cell_partitions(rowmesh, colmesh)

    row_dofs = LinearIndices((ndofs[1], length(rowmesh)))
    col_dofs = LinearIndices((ndofs[2], length(colmesh)))
    I, J = Int[], Int[]
    for (rowcell, colcell) in zip(cells(rowmesh), cells(colmesh))
        append_dofs!(I, J, row_dofs[:, supportnodes(rowmesh, rowcell)], col_dofs[:, supportnodes(colmesh, colcell)])
    end
    sparse(I, J, zeros(T, length(I)), length(row_dofs), length(col_dofs))
end
create_sparse_matrix(meshes::Tuple{IGAMesh{dim,pdim}, IGAMesh{dim,pdim}}; ndofs::Tuple{Int, Int}) where {dim, pdim} = create_sparse_matrix(Float64, meshes; ndofs)

function create_sparse_matrix(::Type{T}, (mesh1,mesh2)::Tuple{FEMesh, FEMesh}; ndofs::Tuple{Int, Int}) where {T}
    mesh1 === mesh2 && return _create_cell_support_sparse_matrix(T, mesh1, ndofs)
    _reference_cell_family(cellshape(mesh1)) === _reference_cell_family(cellshape(mesh2)) || throw(ArgumentError("FEM meshes must use the same reference-cell family"))
    ncells(mesh1) == ncells(mesh2) || throw(DimensionMismatch("FEM meshes must have the same number of cells"))

    gdofs1 = LinearIndices((ndofs[1], length(mesh1)))
    gdofs2 = LinearIndices((ndofs[2], length(mesh2)))
    primarynodes1 = primarynodes_indices(cellshape(mesh1))
    primarynodes2 = primarynodes_indices(cellshape(mesh2))

    I, J = Int[], Int[]
    for (cell1, cell2) in zip(cells(mesh1), cells(mesh2))
        cellnodes1 = supportnodes(mesh1, cell1)
        cellnodes2 = supportnodes(mesh2, cell2)
        mesh1[cellnodes1[primarynodes1]] ≈ mesh2[cellnodes2[primarynodes2]] || throw(ArgumentError("FEM meshes must describe the same cells in the same order and orientation; cell $cell1 does not match"))
        append_dofs!(I, J, gdofs1[:, cellnodes1], gdofs2[:, cellnodes2])
    end

    sparse(I, J, zeros(T, length(I)), length(gdofs1), length(gdofs2))
end
create_sparse_matrix(meshes::Tuple{FEMesh, FEMesh}; ndofs::Tuple{Int, Int}) = create_sparse_matrix(Float64, meshes; ndofs)
create_sparse_matrix(::Type{T}, mesh::FEMesh{<: Any, dim}; ndofs::Int) where {T, dim} = create_sparse_matrix(T, (mesh,mesh); ndofs=(ndofs,ndofs))
create_sparse_matrix(mesh::FEMesh{<: Any, dim}; ndofs::Int) where {dim} = create_sparse_matrix(Float64, mesh; ndofs)

"""
    extract(matrix::AbstractMatrix, dofmap_row::DofMap, dofmap_col::DofMap = dofmap_row)

Extract the active degrees of freedom of a matrix.
"""
function extract(S::AbstractMatrix, dofmap_i, dofmap_j = dofmap_i)
    I, J = _indices_for_extract(S, dofmap_i, dofmap_j)
    S[I, J]
end
function extract(::typeof(view), S::AbstractMatrix, dofmap_i, dofmap_j = dofmap_i)
    I, J = _indices_for_extract(S, dofmap_i, dofmap_j)
    view(S, I, J)
end
function _indices_for_extract(S::AbstractMatrix, dofmap_i::Union{DofMap, Colon}, dofmap_j::Union{DofMap, Colon})
    dofmap_i isa DofMap && @assert size(S, 1) == prod(dofmap_i.masksize)
    dofmap_j isa DofMap && @assert size(S, 2) == prod(dofmap_j.masksize)
    I = dofs(dofmap_i)
    J = dofs(dofmap_j)
    I, J
end
dofs(dofmap::DofMap) = LinearIndices(dofmap.masksize)[dofmap.indices]
dofs(colon::Colon) = colon

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
    size(value) == (row_ndofs, col_ndofs) ||
        throw(DimensionMismatch("matrix value is incompatible with matrix entry dimensions"))
end

# ---- CartesianSparseMatrixAssembler ----

struct CartesianSparseMatrixAssembler{A <: SparseMatrixCSC, R <: LinearIndices, C <: LinearIndices}
    matrix::A
    row_dof_table::R
    col_dof_table::C
    sparsity_radius::Int
end

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
    assembler = CartesianSparseMatrixAssembler(A, row_dof_table, col_dof_table, sparsity_radius)
    has_cartesian_sparse_pattern(assembler) || throw(ArgumentError("Cartesian sparse matrix must use the canonical sparsity pattern"))
    assembler
end

function CartesianSparseMatrixAssembler(A::SparseMatrixCSC, row_mesh::CartesianMesh{N}, col_mesh::CartesianMesh{N}, row_basis::Basis, col_basis::Basis) where {N}
    size(row_mesh) == size(col_mesh) || throw(DimensionMismatch("row and column meshes must have the same size"))
    row_sparsity_radius = support_width(row_basis) - 1
    col_sparsity_radius = support_width(col_basis) - 1
    row_sparsity_radius == col_sparsity_radius || throw(ArgumentError("row and column bases must have the same support width"))
    CartesianSparseMatrixAssembler(A, size(row_mesh), row_sparsity_radius)
end

function cartesian_neighbor_nodes(node::CartesianIndex{N}, mesh_size::Dims{N}, sparsity_radius::Int) where {N}
    shift = sparsity_radius * oneunit(CartesianIndex{N})
    ((node-shift) : (node+shift)) ∩ CartesianIndices(mesh_size)
end

function has_cartesian_sparse_pattern(assembler::CartesianSparseMatrixAssembler)
    (; matrix, row_dof_table, col_dof_table, sparsity_radius) = assembler
    mesh_size = Base.tail(size(row_dof_table))
    row_dofs_per_node = size(row_dof_table, 1)
    col_dofs_per_node = size(col_dof_table, 1)
    rows = rowvals(matrix)
    for col_node in CartesianIndices(mesh_size), b in 1:col_dofs_per_node
        col = col_dof_table[b, col_node]
        indices = nzrange(matrix, col)
        k = first(indices)
        stop = last(indices) + 1
        for row_node in cartesian_neighbor_nodes(col_node, mesh_size, sparsity_radius), a in 1:row_dofs_per_node
            k < stop || return false
            rows[k] == row_dof_table[a, row_node] || return false
            k += 1
        end
        k == stop || return false
    end
    true
end

# The canonical Cartesian CSC pattern allows its destination slots to be computed directly.
@inline function add_entry!(assembler::CartesianSparseMatrixAssembler, row_node::CartesianIndex, col_node::CartesianIndex, value)
    @boundscheck check_cartesian_matrix_entry(assembler, row_node, col_node, value)

    (; matrix, row_dof_table, col_dof_table, sparsity_radius) = assembler
    mesh_size = Base.tail(size(row_dof_table))
    row_ndofs = size(row_dof_table, 1)
    col_ndofs = size(col_dof_table, 1)
    neighboring_nodes = cartesian_neighbor_nodes(col_node, mesh_size, sparsity_radius)

    neighboring_node = row_node - first(neighboring_nodes) + oneunit(row_node)
    row_offset = (LinearIndices(neighboring_nodes)[neighboring_node] - 1) * row_ndofs
    values = nonzeros(matrix)

    @inbounds for b in 1:col_ndofs
        col = col_dof_table[b, col_node]
        slot = first(nzrange(matrix, col)) + row_offset
        for a in 1:row_ndofs
            values[slot] += value[(b - 1) * row_ndofs + a]
            slot += 1
        end
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

# ---- GenericMatrixAssembler ----

struct GenericMatrixAssembler{A <: AbstractMatrix, R <: LinearIndices, C <: LinearIndices}
    matrix::A
    row_dof_table::R
    col_dof_table::C
end

function add!(assembler::GenericMatrixAssembler, row_nodes, col_nodes, local_matrix)
    @_propagate_inbounds_meta
    (; matrix, row_dof_table, col_dof_table) = assembler
    row_dofs, col_dofs = support_dofs(row_dof_table, row_nodes, col_dof_table, col_nodes)
    add!(matrix, row_dofs, col_dofs, local_matrix)
end

# GenericMatrixAssembler writes entries through the matrix indexing interface.
@inline function add_entry!(assembler::GenericMatrixAssembler, row_node, col_node, value)
    (; matrix, row_dof_table, col_dof_table) = assembler
    row_ndofs = size(row_dof_table, 1)
    col_ndofs = size(col_dof_table, 1)
    @boundscheck check_matrix_entry_size(value, row_ndofs, col_ndofs)
    @inbounds for b in 1:col_ndofs, a in 1:row_ndofs
        matrix[row_dof_table[a,row_node], col_dof_table[b,col_node]] += value[(b - 1) * row_ndofs + a]
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
function matrix_assembler(matrix::SparseMatrixCSC, row_mesh::CartesianMesh, col_mesh::CartesianMesh, row_basis::Basis, col_basis::Basis)
    CartesianSparseMatrixAssembler(matrix, row_mesh, col_mesh, row_basis, col_basis)
end

function matrix_dof_tables(gmat, row_grid, col_grid)
    row_table = LinearIndices((size(gmat, 1) ÷ length(row_grid), size(row_grid)...))
    col_table = LinearIndices((size(gmat, 2) ÷ length(col_grid), size(col_grid)...))
    @assert size(gmat) == (length(row_table), length(col_table))
    row_table, col_table
end

# -----------------------------------------------------------------------------
#  LocalMatrixBuffer
# -----------------------------------------------------------------------------

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

function local_dofs(ndofs::Int, index::Integer)
    @_propagate_inbounds_meta
    vec(view(LinearIndices((ndofs, index)), :, index))
end

matrix_row_dofs(buffer::LocalMatrixBuffer, ip) = (@_propagate_inbounds_meta; local_dofs(buffer.row_ndofs, ip))
matrix_col_dofs(buffer::LocalMatrixBuffer, jp) = (@_propagate_inbounds_meta; local_dofs(buffer.col_ndofs, jp))
matrix_row_dofs(::Nothing, ip) = nothing
matrix_col_dofs(::Nothing, jp) = nothing

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

function assemble_add!(assembler, ::Nothing, orientation, row_node, col_node, I, J, value)
    @_propagate_inbounds_meta
    row_node, col_node = orientation((row_node, col_node))
    add_entry!(assembler, row_node, col_node, orient_matrix_entry(orientation, value))
end

finish_assembly!(assembler, ::Nothing, orientation, row_nodes, col_nodes) = assembler.matrix

# -----------------------------------------------------------------------------
#  BlockMatrixBuffer
# -----------------------------------------------------------------------------

struct BlockAssembly{I <: CartesianIndices}
    nodes_i::I
    nodes_j::I
    matrix_buffer_pool::BlockMatrixBufferPool
end

struct BlockMatrixBufferKey{T,N}
    row_size::Dims{N}
    col_size::Dims{N}
    col_offset::CartesianIndex{N}
    row_ndofs::Int
    col_ndofs::Int
    sparsity_radius::Int
end

struct BlockMatrixBuffer{T,N}
    values::Vector{T}
    node_colstarts::Vector{Int}
    key::BlockMatrixBufferKey{T,N}
end

function BlockMatrixBufferKey(assembler::CartesianSparseMatrixAssembler, row_nodes::CartesianIndices{N}, col_nodes::CartesianIndices{N}) where {N}
    (; matrix, row_dof_table, col_dof_table, sparsity_radius) = assembler
    BlockMatrixBufferKey{eltype(matrix),N}(
        size(row_nodes),
        size(col_nodes),
        first(col_nodes) - first(row_nodes),
        size(row_dof_table, 1),
        size(col_dof_table, 1),
        sparsity_radius,
    )
end

function BlockMatrixBuffer(key::BlockMatrixBufferKey{T,N}) where {T,N}
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

function acquire!(pool::BlockMatrixBufferPool, key::BlockMatrixBufferKey{T,N}) where {T,N}
    buffer = lock(pool.lock) do
        buffers = get!(Vector{Any}, pool.buffers, key)
        isempty(buffers) ? nothing : pop!(buffers)
    end
    buffer === nothing && return BlockMatrixBuffer(key)
    buffer = buffer::BlockMatrixBuffer{T,N}
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

# Canonical Cartesian CSC matrices use a block buffer. Matrices using
# GenericMatrixAssembler are written directly within the thread-safe block schedule.
function block_matrix_buffer(assembler::CartesianSparseMatrixAssembler, assembly::BlockAssembly, orientation)
    row_nodes, col_nodes = orientation((assembly.nodes_i, assembly.nodes_j))
    acquire!(assembly.matrix_buffer_pool, BlockMatrixBufferKey(assembler, row_nodes, col_nodes))
end

block_matrix_buffer(::GenericMatrixAssembler, ::BlockAssembly, orientation) = nothing

# ---- assembly ----

function add_entry!(buffer::BlockMatrixBuffer{T,N}, row_nodes::CartesianIndices{N}, col_nodes::CartesianIndices{N}, row_node::CartesianIndex{N}, col_node::CartesianIndex{N}, value) where {T,N}
    @_propagate_inbounds_meta
    @boundscheck check_block_matrix_entry(buffer, row_nodes, col_nodes, row_node, col_node, value)

    (; values, node_colstarts, key) = buffer
    (; row_size, col_size, col_offset, row_ndofs, col_ndofs, sparsity_radius) = key

    local_row_node = row_node - first(row_nodes) + oneunit(row_node)
    local_col_node = col_node - first(col_nodes) + oneunit(col_node)
    neighboring_rows = cartesian_neighbor_nodes(local_col_node + col_offset, row_size, sparsity_radius)
    neighboring_row = local_row_node - first(neighboring_rows) + oneunit(local_row_node)
    local_row = LinearIndices(neighboring_rows)[neighboring_row]
    local_col = LinearIndices(col_size)[local_col_node]

    slot = node_colstarts[local_col] + (local_row - 1) * row_ndofs * col_ndofs
    for index in eachindex(value)
        values[slot] += value[index]
        slot += 1
    end

    buffer
end

@noinline function check_block_matrix_entry(buffer::BlockMatrixBuffer{T,N}, row_nodes::CartesianIndices{N}, col_nodes::CartesianIndices{N}, row_node::CartesianIndex{N}, col_node::CartesianIndex{N}, value) where {T,N}
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

function scatter!(assembler::CartesianSparseMatrixAssembler, buffer::BlockMatrixBuffer{T,N}, row_nodes::CartesianIndices{N}, col_nodes::CartesianIndices{N}) where {T,N}
    @_propagate_inbounds_meta
    @boundscheck check_block_matrix_scatter(assembler, buffer, row_nodes, col_nodes)

    (; matrix, row_dof_table, col_dof_table, sparsity_radius) = assembler
    (; values, node_colstarts, key) = buffer
    (; row_size, col_size, col_offset, row_ndofs, col_ndofs) = key

    mesh_size = Base.tail(size(row_dof_table))
    matrix_values = nonzeros(matrix)
    first_row_node = first(row_nodes)
    first_col_node = first(col_nodes)
    for (local_col, local_col_node) in enumerate(CartesianIndices(col_size))
        col_node = local_col_node + first_col_node - oneunit(first_col_node)
        neighboring_rows = cartesian_neighbor_nodes(local_col_node + col_offset, row_size, sparsity_radius)
        matrix_neighboring_rows = cartesian_neighbor_nodes(col_node, mesh_size, sparsity_radius)
        for b in 1:col_ndofs
            col = col_dof_table[b,col_node]
            matrix_col_start = first(nzrange(matrix, col))
            for (local_row, local_row_node) in enumerate(neighboring_rows)
                local_slot = node_colstarts[local_col] + ((local_row - 1) * col_ndofs + b - 1) * row_ndofs
                row_node = local_row_node + first_row_node - oneunit(first_row_node)
                matrix_neighboring_row = row_node - first(matrix_neighboring_rows) + oneunit(row_node)
                matrix_slot = matrix_col_start + (LinearIndices(matrix_neighboring_rows)[matrix_neighboring_row] - 1) * row_ndofs
                for _ in 1:row_ndofs
                    matrix_values[matrix_slot] += values[local_slot]
                    matrix_slot += 1
                    local_slot += 1
                end
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

# Canonical Cartesian CSC entries are accumulated in the block buffer.
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

# -----------------------------------------------------------------------------
#  P2G_Matrix
# -----------------------------------------------------------------------------

struct ParticleAssembly end
struct CellAssembly end

# ---- support nodes ----

@inline function matrix_supportnodes(bw, grid)
    @_propagate_inbounds_meta
    # Matrix assembly indexes global DOF tables, which are built on logical
    # grid indices. For an SpGrid, supportnodes(bw, grid) returns SpIndex
    # storage tokens instead. Using those here would require SpIndex to fully
    # support AbstractArray indexing.
    nodes = supportnodes(bw)
    @boundscheck checkbounds(get_mesh(grid), nodes)
    nodes, nodes
end

@inline function matrix_supportnodes(bw_i, grid_i, bw_j, grid_j)
    @_propagate_inbounds_meta
    # See the single-grid method: matrix DOF tables need logical grid indices,
    # not SpGrid storage tokens.
    nodes_i = supportnodes(bw_i)
    nodes_j = supportnodes(bw_j)
    @boundscheck checkbounds(get_mesh(grid_i), nodes_i)
    @boundscheck checkbounds(get_mesh(grid_j), nodes_j)
    nodes_i, nodes_j
end

function matrix_block_supportnodes(weights, particle_indices, grid)
    @_propagate_inbounds_meta
    p, remaining_particles = Iterators.peel(particle_indices)
    nodes = supportnodes(weights[p])
    first_node = first(nodes)
    last_node = last(nodes)
    for p in remaining_particles
        nodes = supportnodes(weights[p])
        first_node = CartesianIndex(map(min, Tuple(first_node), Tuple(first(nodes))))
        last_node = CartesianIndex(map(max, Tuple(last_node), Tuple(last(nodes))))
    end
    nodes = first_node:last_node
    @boundscheck checkbounds(get_mesh(grid), nodes)
    nodes
end

# ---- assembly scheduling ----

function P2G_Matrix(f, device::CPUDevice, schedule::Val, grids, particles, weights, partition)
    P2G((grids, particles, weights, p) -> (@inline f(grids, particles, weights, (p,), ParticleAssembly())), device, schedule, grids, particles, weights, partition)
end

function P2G_Matrix(f, ::CPUDevice, ::Val{scheduler}, grids, particles, weights, partition::ThreadPartition{<:BlockStrategy}) where {scheduler}
    matrix_buffer_pool = strategy(partition).matrix_buffer_pool
    for group in threadsafe_groups(partition)
        tforeach(group, scheduler) do block
            block_particle_indices = particle_indices(partition, particles, block)
            nodes_i = matrix_block_supportnodes(weights[1], block_particle_indices, grids[1])
            nodes_j = grids[1] === grids[2] && weights[1] === weights[2] ? nodes_i : matrix_block_supportnodes(weights[2], block_particle_indices, grids[2])
            @inline f(grids, particles, weights, block_particle_indices, BlockAssembly(nodes_i, nodes_j, matrix_buffer_pool))
        end
    end
end

function P2G_Matrix(f, ::CPUDevice, ::Val{scheduler}, grids, particles::QuadraturePoints,
                    weights::Tuple{<:BasisWeightArray{<:Any, <:Any, <:CellSupportMatrix}, <:BasisWeightArray{<:Any, <:Any, <:CellSupportMatrix}},
                    ::Nothing) where {scheduler}
    scheduler == :nothing || @warn "@P2G_Matrix: `ThreadPartition` must be given for threaded computation" maxlog=1

    for cell in axes(particles, 2)
        particle_indices = (CartesianIndex(q, cell) for q in axes(particles, 1))
        @inline f(grids, particles, weights, particle_indices, CellAssembly())
    end
end

function P2G_Matrix(f, ::CPUDevice, ::Val{scheduler}, grids, particles::QuadraturePoints,
                    weights::Tuple{<:BasisWeightArray{<:Any, <:Any, <:CellSupportMatrix}, <:BasisWeightArray{<:Any, <:Any, <:CellSupportMatrix}},
                    partition::ThreadPartition{<:CellStrategy}) where {scheduler}
    for group in threadsafe_groups(partition)
        tforeach(group, scheduler) do cell
            @inline f(grids, particles, weights, particle_indices(partition, particles, cell), CellAssembly())
        end
    end
end

function check_arguments_for_P2G_Matrix(grid, particles, weights, partition)
    check_arguments_for_P2G(grid, particles, weights, partition)
    @assert get_device(grid) isa CPUDevice
end

# ---- macro implementation ----

"""
    @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) [partition] begin
        equations...
    end

Particle-to-grid transfer macro for assembling a global matrix.
A typical global stiffness matrix can be assembled as follows:

```julia
@P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
    K[i,j] = @∑ ∇w[ip] ⊡ c[p] ⊡ ∇w[jp] * V[p]
end
```

where `c` and `V` denote the stiffness (symmetric fourth-order) tensor and the volume, respectively.
It is recommended to create global stiffness `K` using [`create_sparse_matrix`](@ref).
"""
macro P2G_Matrix(grid_ij, particles_p, weights_ipjp, equations)
    P2G_Matrix_expr(QuoteNode(:nothing), grid_ij, particles_p, weights_ipjp, nothing, equations)
end
macro P2G_Matrix(grid_ij, particles_p, weights_ipjp, partition, equations)
    P2G_Matrix_expr(QuoteNode(:nothing), grid_ij, particles_p, weights_ipjp, partition, equations)
end
macro P2G_Matrix(schedule::QuoteNode, grid_ij, particles_p, weights_ipjp, equations)
    P2G_Matrix_expr(schedule, grid_ij, particles_p, weights_ipjp, nothing, equations)
end
macro P2G_Matrix(schedule::QuoteNode, grid_ij, particles_p, weights_ipjp, partition, equations)
    P2G_Matrix_expr(schedule, grid_ij, particles_p, weights_ipjp, partition, equations)
end

function P2G_Matrix_expr(schedule, grid_ij, particles_p, weights_ipjp, partition, equations)
    P2G_Matrix_expr(schedule, unpair2(grid_ij), unpair(particles_p), unpair2(weights_ipjp), partition, parse_transfer_program(equations))
end

function P2G_Matrix_expr(schedule::QuoteNode, ((grid_i,grid_j),(i,j)), (particles,p), ((weights_i,weights_j),(ip,jp)), partition, program::TransferProgram)
    @gensym grid_i′ grid_j′ weights_i′ weights_j′ bw_i bw_j gridindices_i gridindices_j particle_indices matrix_assembly remaining_particles

    equations = program.equations
    isempty(equations) && error("@P2G_Matrix: at least one equation is required")
    all(is_sum, equations) || error("@P2G_Matrix: all equations must use `@∑`")

    scope = TransferScope([grid_i′=>i, grid_j′=>j, particles=>p, bw_i=>ip, bw_j=>jp]; cache=true)
    equations = map(equations) do eq
        TransferEquation(eq.kind, eq.lhs, resolve_refs(eq.rhs, scope), eq.op)
    end
    particle_replacements = cached_replacements(scope, p)
    i_replacements = cached_replacements(scope, i, ip)
    j_replacements = cached_replacements(scope, j, jp)
    inner_symbols = p2g_cached_symbols(cached_replacements(scope, i, j, ip, jp))

    gmats = Any[]
    matrices_init = Any[]
    hoist_exprs = Any[]
    buffers_init = Any[]
    block_buffers_init = Any[]
    local_jdofs = Any[]
    local_idofs = Any[]
    assemble_first = Any[]
    assemble_add = Any[]
    assemble_block_entries = Any[]
    finish_assemblies = Any[]
    finish_block_assemblies = Any[]
    for equation in equations
        (; lhs, rhs, op) = equation
        @capture(lhs, gmat_[gi_,gj_]) || error("@P2G_Matrix: Invalid global matrix expression, got `$lhs`")
        ((gi == i && gj == j) || (gi == j && gj == i)) || error("@P2G_Matrix: Expected expression of the form `$gmat[$i, $j]` or `$gmat[$j, $i]`, got `$lhs`")
        gmat in gmats && error("@P2G_Matrix: each global matrix may appear only once in a block; combine terms for `$gmat` into one `@∑` expression")

        buffer = gensym(Symbol(gmat, :buffer))
        assembler = gensym(Symbol(gmat, :assembler))
        I = gensym(Symbol(gmat, :I))
        J = gensym(Symbol(gmat, :J))

        fillzero = op == :(=) ? :(Tesserae.fillzero!($gmat)) : nothing
        op == :(-=) && (rhs = :(-$rhs))
        rhs = hoist_p2g_rhs!(hoist_exprs, inner_symbols, rhs)
        push!(gmats, gmat)
        forward = gi == i && gj == j
        reorder_pair = forward ? identity : reverse
        orientation = forward ? :(Base.identity) : :(Base.reverse)
        row_grid, col_grid = reorder_pair((grid_i, grid_j))
        row_weights, col_weights = reorder_pair((weights_i, weights_j))
        dof_table_i, dof_table_j = reorder_pair((:($(assembler).row_dof_table), :($(assembler).col_dof_table)))
        matrix_cache = Symbol(gmat, :cache)
        push!(matrices_init, quote
            $fillzero
            $assembler = Tesserae.matrix_assembler($gmat, Tesserae.get_mesh($row_grid), Tesserae.get_mesh($col_grid), Tesserae.basis($row_weights), Tesserae.basis($col_weights))
            $matrix_cache = Tesserae.local_matrix_cache($gmat, $dof_table_i, $weights_i, $dof_table_j, $weights_j)
        end)
        push!(buffers_init, quote
            if $matrix_assembly isa Tesserae.ParticleAssembly
                $buffer = nothing
            elseif $matrix_assembly isa Tesserae.CellAssembly
                $buffer = Tesserae.local_matrix_buffer($matrix_cache, $dof_table_i, $gridindices_i, $dof_table_j, $gridindices_j)
            else
                error("BlockAssembly must use block assembly")
            end
        end)
        push!(block_buffers_init, :($buffer = Tesserae.block_matrix_buffer($assembler, $matrix_assembly, $orientation)))
        push!(local_jdofs, :($J = Tesserae.matrix_col_dofs($buffer, $jp)))
        push!(local_idofs, :($I = Tesserae.matrix_row_dofs($buffer, $ip)))
        push!(assemble_first, :(Tesserae.assemble_first!($assembler, $buffer, $orientation, $i, $j, $I, $J, $rhs)))
        push!(assemble_add, :(Tesserae.assemble_add!($assembler, $buffer, $orientation, $i, $j, $I, $J, $rhs)))
        push!(assemble_block_entries, :(Tesserae.assemble_block_entry!($assembler, $buffer, $matrix_assembly, $orientation, $i, $j, $rhs)))
        push!(finish_assemblies, :(Tesserae.finish_assembly!($assembler, $buffer, $orientation, $gridindices_i, $gridindices_j)))
        push!(finish_block_assemblies, :(Tesserae.finish_block_assembly!($assembler, $buffer, $matrix_assembly, $orientation)))
    end

    supportnodes_expr = if grid_i == grid_j && weights_i == weights_j
        :(($gridindices_i, $gridindices_j) = Tesserae.matrix_supportnodes($bw_i, $grid_i′))
    else
        :(($gridindices_i, $gridindices_j) = Tesserae.matrix_supportnodes($bw_i, $grid_i′, $bw_j, $grid_j′))
    end

    particle_init = quote
        $(particle_replacements...)
        $(hoist_exprs...)
        $bw_i, $bw_j = $weights_i′[$p], $weights_j′[$p]
    end

    function assemble_particle(assembly)
        quote
            for $jp in eachindex($gridindices_j)
                $j = $gridindices_j[$jp]
                $(j_replacements...)
                $(local_jdofs...)
                for $ip in eachindex($gridindices_i)
                    $i = $gridindices_i[$ip]
                    $(i_replacements...)
                    $(local_idofs...)
                    $(assembly...)
                end
            end
        end
    end

    particle_or_cell_body = quote
        $p, $remaining_particles = Base.Iterators.peel($particle_indices)
        $particle_init
        $supportnodes_expr
        $(buffers_init...)
        $(assemble_particle(assemble_first))
        for $p in $remaining_particles
            $particle_init
            $(assemble_particle(assemble_add))
        end
        $(finish_assemblies...)
    end

    block_body = quote
        $(block_buffers_init...)
        for $p in $particle_indices
            $particle_init
            $supportnodes_expr
            for $jp in eachindex($gridindices_j)
                $j = $gridindices_j[$jp]
                $(j_replacements...)
                for $ip in eachindex($gridindices_i)
                    $i = $gridindices_i[$ip]
                    $(i_replacements...)
                    $(assemble_block_entries...)
                end
            end
        end
        $(finish_block_assemblies...)
    end

    body = quote
        if $matrix_assembly isa Tesserae.BlockAssembly
            $block_body
        else
            $particle_or_cell_body
        end
    end

    if !DEBUG
        body = :(@inbounds $body)
    end

    body = quote
        let
            $check_arguments_for_P2G_Matrix($grid_i, $particles, $weights_i, $partition)
            $check_arguments_for_P2G_Matrix($grid_j, $particles, $weights_j, $partition)
            $(matrices_init...)
            Tesserae.P2G_Matrix((($grid_i′,$grid_j′), $particles, ($weights_i′,$weights_j′), $particle_indices, $matrix_assembly) -> $body,
                                $get_device($grid_i), Val($schedule), ($grid_i,$grid_j), $particles, ($weights_i,$weights_j), $partition)
        end
    end

    esc(interpolate_transfer_values(body, program))
end

function unpair2(ex::Expr)
    if @capture(ex, lhs_Symbol => (rhs1_Symbol, rhs2_Symbol))
        return (lhs, lhs), (rhs1, rhs2)
    elseif @capture(ex, (lhs1_Symbol, lhs2_Symbol) => (rhs1_Symbol, rhs2_Symbol))
        return (lhs1, lhs2), (rhs1, rhs2)
    else
        error("invalid expression, $ex")
    end
end

"""
    Tesserae.newton!(x::AbstractVector, f, J,
                     maxiter = 100, atol = zero(eltype(x)), rtol = sqrt(eps(eltype(x))),
                     linsolve = (x,A,b) -> copyto!(x, A\\b),
                     backtracking = false, verbose = false)

A simple implementation of Newton's method.
The functions `f(x)` and `J(x)` should return the residual vector and its Jacobian, respectively.

Evaluation order:

```julia
r = f(x)              # update state/caches derived from x and return residual
while not converged
    x_old = x
    Jx = J(x)         # compute from x or reuse caches from f(x)
    δx = solve(Jx, r)

    if backtracking
        ϕ′0 = -dot(r, Jx, δx)
        ϕ′0 < 0 || fail
        for α in trial_steps
            x = x_old - α * δx
            r = f(x)  # update trial state
            accept && break
        end
    else
        x = x_old - δx
        r = f(x)
    end
end
```

If backtracking fails, `x` is restored to the last accepted iterate and `f(x)` is called once more to restore the corresponding state.

!!! tip
    At each iteration, `newton!` evaluates `J(x)` only after `f(x)` has already been evaluated at the same `x`.
    In simulation codes, residual and tangent/Jacobian assembly often share intermediate quantities.
    These quantities may be stored in caller-owned state while evaluating `f(x)`, so that the following `J(x)` call can reuse them without recomputing them.
    This is optional: `J(x)` may also assemble the Jacobian directly from `x`.
"""
function newton!(
        x::AbstractVector, f, J;
        maxiter::Int=100, atol::Real=zero(eltype(x)), rtol::Real=sqrt(eps(eltype(x))),
        linsolve=(x,A,b)->copyto!(x,A\b), backtracking::Bool=false, verbose::Bool=false)

    T = eltype(x)

    r = f(x)
    rnorm = rnorm0 = norm(r)
    δx = similar(x)

    # old accepted step values
    x_old, rnorm_old = similar(x), rnorm

    iter = 0
    solved = rnorm0 ≤ atol
    giveup = !isfinite(rnorm)

    if verbose
        newton_print_header(maxiter, atol, rtol)
        newton_print_row(maxiter, iter, rnorm, newton_residual_ratio(rnorm, rnorm0))
    end

    while !(solved || giveup)
        @. x_old = x
        rnorm_old = rnorm

        Jx = J(x)
        linsolve(fillzero!(δx), Jx, r)

        if backtracking
            ϕ0 = rnorm_old * rnorm_old / 2
            ϕ′0 = -dot(r, Jx, δx)
            if !(isfinite(ϕ′0) && ϕ′0 < 0)
                giveup = true
                break
            end
            accepted = newton_backtracking(one(T), ϕ0, ϕ′0) do α
                @. x = x_old - α * δx # update `x`
                r .= f(x) # update r in backtracking process
                y = norm(r)
                y * y / 2
            end
            if !accepted
                @. x = x_old
                f(x) # restore state derived from x_old
                giveup = true
                break
            end
        else
            @. x = x_old - δx
            r .= f(x)
        end

        rnorm = norm(r)
        solved = rnorm ≤ max(atol, rtol*rnorm0)
        iter += 1
        giveup = !isfinite(rnorm) || iter ≥ maxiter

        verbose && newton_print_row(maxiter, iter, rnorm, newton_residual_ratio(rnorm, rnorm0))
    end
    verbose && println()

    solved
end

newton_residual_ratio(rnorm, rnorm0) = iszero(rnorm0) ? zero(rnorm0) : rnorm/rnorm0

function newton_print_header(maxiter, atol, rtol)
    n = ndigits(maxiter)
    @printf(" # ≤ %d  f ≤ %-8.2e  f/f₀ ≤ %-8.2e\n", maxiter, atol, rtol)
    @printf(" %s  %s  %s\n", "─"^(4+n), "─"^12, "─"^15)
end
function newton_print_row(maxiter, iter, f, f_f0)
    n = ndigits(maxiter)
    @printf(" %s%s  %12.2e  %15.2e\n", " "^4, lpad(iter, n), f, f_f0)
end

function newton_backtracking(ϕ, α::T, ϕ0::T, ϕ′0::T; c::T = T(1e-4), ρ_hi::T = T(0.5), ρ_lo::T = T(0.1), maxiter::Int=1000) where {T <: Real}
    @assert 0 < ρ_lo < ρ_hi < 1
    local α_prev, ϕα_prev
    for trial in 1:maxiter
        ϕα = ϕ(α)
        ϕα ≤ ϕ0 + c*α*ϕ′0 && return true
        abs(α) < eps(T)^T(2/3) && return false

        if trial == 1
            α_new = quad_step(α, ϕα, ϕ0, ϕ′0, ρ_hi, ρ_lo)
        else
            α_new = cubic_step(α, ϕα, α_prev, ϕα_prev, ϕ0, ϕ′0, ρ_hi, ρ_lo)
        end
        α_new = clamp(α_new, α*ρ_lo, α*ρ_hi)
        α_prev, ϕα_prev = α, ϕα
        α = α_new
    end
    false
end

function quad_step(α, ϕα, ϕ0, ϕ′0, ρ_hi, ρ_lo)
    den = 2(ϕα - α*ϕ′0 - ϕ0)
    if isfinite(den) && den > 0
        return -α^2 * ϕ′0 / den
    else
        return ρ_lo * α
    end
end

function cubic_step(α, ϕα, α_prev, ϕα_prev, ϕ0, ϕ′0, ρ_hi, ρ_lo)
    den = α_prev^2 * α^2 * (α - α_prev)
    if isfinite(den) && !iszero(den)
        sα = ϕα - ϕ0 - ϕ′0*α
        sα_prev = ϕα_prev - ϕ0 - ϕ′0*α_prev
        a = ( α_prev^2 * sα - α^2 * sα_prev) / den
        b = (-α_prev^3 * sα + α^3 * sα_prev) / den

        !(isfinite(a) && isfinite(b)) && return ρ_lo * α

        # quadratic
        if abs(a) ≤ eps(typeof(a)) && !iszero(b)
            return -ϕ′0 / 2b
        end

        # cubic
        d = b^2 - 3a*ϕ′0
        if isfinite(d) && d ≥ 0 && !iszero(a)
            α_new = (-b + sqrt(d)) / 3a
            isfinite(α_new) && α_new > 0 && return α_new
        end
    end
    ρ_lo * α
end
