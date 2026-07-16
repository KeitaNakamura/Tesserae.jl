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
    create_sparse_matrix(basis, mesh; ndofs)

Create a sparse matrix.
Since the created matrix accounts for all nodes in the mesh,
it needs to be extracted for active nodes using the `DofMap`.
`ndofs` specifies the number of DoFs for a field and must be provided explicitly.

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
    @gensym grid_i′ grid_j′ weights_i′ weights_j′ bw_i bw_j gridindices_i gridindices_j

    equations = program.equations
    isempty(equations) && error("@P2G_Matrix: at least one equation is required")
    all(is_sum, equations) || error("@P2G_Matrix: all equations must use `@∑`")

    scope = TransferScope([grid_i′=>i, grid_j′=>j, particles=>p, bw_i=>ip, bw_j=>jp]; cache=true)
    equations = map(equations) do eq
        TransferEquation(eq.kind, eq.lhs, resolve_refs(eq.rhs, scope), eq.op)
    end
    replaced = scope.replacements
    inner_symbols = p2g_cached_symbols(replaced, 1, 2, 4, 5)

    fillzeros = Any[]
    gmats = Any[]
    gdofs_init = Any[]
    hoist_exprs = Any[]
    lmat_init = Any[]
    local_jdofs = Any[]
    local_idofs = Any[]
    lmat_asm = Any[]
    gdofs_extract = Any[]
    lmat2gmat = Any[]
    for k in eachindex(equations)
        (; lhs, rhs, op) = equations[k]
        @capture(lhs, gmat_[gi_,gj_]) || error("@P2G_Matrix: Invalid global matrix expression, got `$lhs`")
        ((gi == i && gj == j) || (gi == j && gj == i)) || error("@P2G_Matrix: Expected expression of the form `$gmat[$i, $j]` or `$gmat[$j, $i]`, got `$lhs`")
        gmat in gmats && error("@P2G_Matrix: each global matrix may appear only once in a block; combine terms for `$gmat` into one `@∑` expression")

        lmat = gensym(gmat)
        gdofs_i = gensym(Symbol(gmat, :gdofs_i))
        gdofs_j = gensym(Symbol(gmat, :gdofs_j))
        ldofs_i = gensym(Symbol(gmat, :ldofs_i))
        ldofs_j = gensym(Symbol(gmat, :ldofs_j))
        dofs_i = gensym(Symbol(gmat, :dofs_i))
        dofs_j = gensym(Symbol(gmat, :dofs_j))
        I = gensym(Symbol(gmat, :I))
        J = gensym(Symbol(gmat, :J))

        op == :(=)  && push!(fillzeros, :(Tesserae.fillzero!($gmat)))
        op == :(-=) && (rhs = :(-$rhs))
        rhs = hoist_p2g_rhs!(hoist_exprs, inner_symbols, rhs)
        lmat_dims = Symbol(lmat, :dims)
        push!(gmats, gmat)
        if gi == i && gj == j
            push!(gdofs_init, :(($gdofs_i, $gdofs_j) = Tesserae.matrix_dof_tables($gmat, $grid_i, $grid_j)))
        else
            push!(gdofs_init, :(($gdofs_j, $gdofs_i) = Tesserae.matrix_dof_tables($gmat, $grid_j, $grid_i)))
        end
        push!(lmat_init, quote
            $ldofs_i = Tesserae.local_dof_table($gdofs_i, $gridindices_i)
            $ldofs_j = Tesserae.local_dof_table($gdofs_j, $gridindices_j)
            $lmat_dims = length($ldofs_i), length($ldofs_j)
            $lmat = get!(()->Array{eltype($gmat)}(undef, $lmat_dims), $(Symbol(gmat,:dict))[], $lmat_dims)
        end)
        push!(local_jdofs, :($J = Tesserae.local_dofs($ldofs_j, $jp)))
        push!(local_idofs, :($I = Tesserae.local_dofs($ldofs_i, $ip)))
        push!(lmat_asm, :(@inbounds $lmat[$I,$J] .= $rhs))
        push!(gdofs_extract, :(($dofs_i, $dofs_j) = Tesserae.support_dofs($gdofs_i, $gridindices_i, $gdofs_j, $gridindices_j)))
        if gi == i && gj == j
            push!(lmat2gmat, :(Tesserae.add!($gmat, $dofs_i, $dofs_j, $lmat)))
        else
            push!(lmat2gmat, :(Tesserae.add!($gmat, $dofs_j, $dofs_i, $lmat')))
        end
    end

    supportnodes_expr = if grid_i == grid_j && weights_i == weights_j
        :(($gridindices_i, $gridindices_j) = Tesserae.matrix_supportnodes($bw_i, $grid_i′))
    else
        :(($gridindices_i, $gridindices_j) = Tesserae.matrix_supportnodes($bw_i, $grid_i′, $bw_j, $grid_j′))
    end

    body = quote
        $(replaced[3]...)
        $(hoist_exprs...)
        $bw_i, $bw_j = $weights_i′[$p], $weights_j′[$p]
        $supportnodes_expr
        $(lmat_init...)
        for $jp in eachindex($gridindices_j)
            $j = $gridindices_j[$jp]
            $(cached_replacements(scope, 2, 5)...)
            $(local_jdofs...)
            for $ip in eachindex($gridindices_i)
                $i = $gridindices_i[$ip]
                $(cached_replacements(scope, 1, 4)...)
                $(local_idofs...)
                $(lmat_asm...)
            end
        end
        $(gdofs_extract...)
        $(lmat2gmat...)
    end

    if !DEBUG
        body = :(@inbounds $body)
    end

    # cache for local matrices
    arraydicts = Any[]
    for gmat in gmats
        arraydict = Symbol(gmat, :dict)
        Tarraydict = Symbol(:T, arraydict)
        ex = quote
            $Tarraydict = Dict{Tuple{Int,Int}, Matrix{eltype($gmat)}}
            $arraydict = $TaskLocalValue{$Tarraydict}(() -> $Tarraydict())
            $arraydict[] # initialize
        end
        push!(arraydicts, ex)
    end

    body = quote
        let
            $(arraydicts...)
            $check_arguments_for_P2G_Matrix($grid_i, $particles, $weights_i, $partition)
            $check_arguments_for_P2G_Matrix($grid_j, $particles, $weights_j, $partition)
            $(fillzeros...)
            $(gdofs_init...)
            $P2G((($grid_i′,$grid_j′), $particles, ($weights_i′,$weights_j′), $p) -> $body, $get_device($grid_i), Val($schedule), ($grid_i,$grid_j), $particles, ($weights_i,$weights_j), $partition)
        end
    end

    esc(interpolate_transfer_values(body, program))
end

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

function matrix_dof_tables(gmat, row_grid, col_grid)
    row_table = LinearIndices((size(gmat, 1) ÷ length(row_grid), size(row_grid)...))
    col_table = LinearIndices((size(gmat, 2) ÷ length(col_grid), size(col_grid)...))
    @assert size(gmat) == (length(row_table), length(col_table))
    row_table, col_table
end

@inline function local_dof_table(dof_table, nodes)
    @_propagate_inbounds_meta
    LinearIndices((size(dof_table, 1), size(nodes)...))
end

@inline function local_dofs(local_table, ip)
    @_propagate_inbounds_meta
    vec(view(local_table, :, ip))
end

@inline function support_dofs(table_i, nodes_i, table_j, nodes_j)
    @_propagate_inbounds_meta
    if size(table_i, 1) == size(table_j, 1) && nodes_i === nodes_j
        dofs = vec(table_i[:, nodes_i])
        return dofs, dofs
    else
        return vec(table_i[:, nodes_i]), vec(table_j[:, nodes_j])
    end
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

function check_arguments_for_P2G_Matrix(grid, particles, weights, partition)
    check_arguments_for_P2G(grid, particles, weights, partition)
    @assert get_device(grid) isa CPUDevice
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
