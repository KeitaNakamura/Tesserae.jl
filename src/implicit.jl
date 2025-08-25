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
    A′ = reshape(reinterpret(eltype(T), A), 1, length(A))
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
    create_sparse_matrix(interpolation, mesh; ndofs = ndims(mesh))

Create a sparse matrix.
Since the created matrix accounts for all nodes in the mesh,
it needs to be extracted for active nodes using the `DofMap`.
`ndofs` represents the number of DoFs for a field.

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
function create_sparse_matrix(interp::Interpolation, mesh::AbstractMesh; ndofs::Int = ndims(mesh))
    create_sparse_matrix(Float64, interp, mesh; ndofs)
end
function create_sparse_matrix(::Type{T}, interp::Interpolation, mesh::CartesianMesh{dim}; ndofs::Int = dim) where {T, dim}
    dims = size(mesh)
    n = ndofs * prod(dims)
    I, J = Int[], Int[]
    LI, CI = LinearIndices(dims), CartesianIndices(dims)
    for i in CI
        unit = (kernel_support(interp) - 1) * oneunit(i)
        indices = intersect((i-unit):(i+unit), CI)
        idofs = (ndofs*(LI[i]-1)+1):(ndofs*LI[i])
        for j in indices
            jdofs = (ndofs*(LI[j]-1)+1):(ndofs*LI[j])
            append_dofs!(I, J, idofs, jdofs)
        end
    end
    sparse(I, J, zeros(T, length(I)), n, n)
end

function append_dofs!(I, J, idofs, jdofs)
    for j in 1:length(jdofs)
        append!(I, idofs)
        for _ in 1:length(idofs)
            push!(J, jdofs[j])
        end
    end
end

function create_sparse_matrix(::Type{T}, meshes::Tuple{Vararg{UnstructuredMesh, N}}; ndofs::NTuple{N, Int}) where {T, N}
    gdofs = generate_dofs(meshes, ndofs)
    ttldofs = sum(length, gdofs)
    I, J = Int[], Int[]
    for i in 1:N
        mesh_i = meshes[i]
        gdofs_i = gdofs[i]
        for c_i in 1:ncells(mesh_i)
            cellnodes_i = cellnodeindices(mesh_i, c_i)
            primarycellnodes_i = cellnodes_i[primarynodes_indices(cellshape(mesh_i))]
            celldofs_i = gdofs_i[:, cellnodes_i]
            for j in 1:N
                if i == j
                    append_dofs!(I, J, celldofs_i, celldofs_i)
                else # check shared cells
                    mesh_j = meshes[j]
                    gdofs_j = gdofs[j]
                    for c_j in 1:ncells(mesh_j)
                        cellnodes_j = cellnodeindices(mesh_j, c_j)
                        primarycellnodes_j = cellnodes_j[primarynodes_indices(cellshape(mesh_j))]
                        if mesh_i[primarycellnodes_i] ≈ mesh_j[primarycellnodes_j] # TODO: better way because this is very slow
                            celldofs_j = gdofs_j[:, cellnodes_j]
                            append_dofs!(I, J, celldofs_i, celldofs_j)
                        end
                    end
                end
            end
        end
    end
    sparse(I, J, zeros(T, length(I)), ttldofs, ttldofs)
end
create_sparse_matrix(meshes::Tuple{Vararg{UnstructuredMesh}}; ndofs::Dims) = create_sparse_matrix(Float64, meshes; ndofs)

function generate_dofs(meshes::Tuple{Vararg{UnstructuredMesh, N}}, ndofs::Tuple{Vararg{Int, N}}) where {N}
    ttldofs = map(i -> ndofs[i]*length(meshes[i]), 1:N)
    ntuple(Val(N)) do i
        offset = sum(ttldofs[1:i-1])
        LinearIndices((ndofs[i], length(meshes[i]))) .+ offset
    end
end

create_sparse_matrix(::Type{T}, mesh::UnstructuredMesh{<: Any, dim}; ndofs::Int = dim) where {T, dim} = create_sparse_matrix(T, (mesh,); ndofs=(ndofs,))
create_sparse_matrix(mesh::UnstructuredMesh{<: Any, dim}; ndofs::Int = dim) where {dim} = create_sparse_matrix(Float64, mesh; ndofs)

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
    I = dofindices(dofmap_i)
    J = dofindices(dofmap_j)
    I, J
end
dofindices(dofmap::DofMap) = LinearIndices(dofmap.masksize)[dofmap.indices]
dofindices(colon::Colon) = colon

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
        nzrng_i = 1
        nzrng = nzrange(A, J[j])
        while nzrng_i in eachindex(nzrng) && i in eachindex(I)
            k = nzrng[nzrng_i]
            row = rows[k] # row candidate
            i′ = perm[i]
            if I[i′] == row
                vals[k] += K[i′,j]
                i += 1
            end
            nzrng_i += 1
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
    P2G_Matrix_expr(schedule, unpair2(grid_ij), unpair(particles_p), unpair2(weights_ipjp), partition, split_equations(equations))
end

function P2G_Matrix_expr(schedule::QuoteNode, ((grid_i,grid_j),(i,j)), (particles,p), ((weights_i,weights_j),(ip,jp)), partition, equations::Vector)
    @gensym grid_i′ grid_j′ weights_i′ weights_j′ iw_i iw_j gridindices_i gridindices_j ldofs_i ldofs_j I J dofs_i dofs_j gdofs_i gdofs_j

    @assert all(eq -> eq.issumeq, equations)

    maps = [grid_i′=>i, grid_j′=>j, particles=>p, iw_i=>ip, iw_j=>jp]
    replaced = [Set{Expr}(), Set{Expr}(), Set{Expr}(), Set{Expr}(), Set{Expr}()]
    for k in eachindex(equations)
        eq = equations[k]
        eq.rhs = resolve_refs(eq.rhs, maps; replaced)
    end

    fillzeros = Any[]
    gmats = Any[]
    lmat_init = Any[]
    lmat_asm = Any[] 
    lmat2gmat = Any[]
    for k in eachindex(equations)
        (; lhs, rhs, op) = equations[k]
        @capture(lhs, gmat_[gi_,gj_]) || error("@P2G_Matrix: Invalid global matrix expression, got `$lhs`")
        (gi == i && gj == j) || error("@P2G_Matrix: Expected expression of the form `$gmat[$i, $j]`, got `$lhs`")
        lmat = gensym(gmat)
        op == :(=)  && push!(fillzeros, :(Tesserae.fillzero!($gmat)))
        op == :(-=) && (rhs = :(-$rhs))
        push!(gmats, gmat)
        push!(lmat_init, :($lmat = Array{eltype($gmat)}(undef, length($ldofs_i), length($ldofs_j))))
        push!(lmat_asm, :(@inbounds $lmat[$I,$J] .= $trySArray($rhs))) # converting `Tensor` to `SArray` is faster for setindex!
        push!(lmat2gmat, :(Tesserae.add!($gmat, $dofs_i, $dofs_j, $lmat)))
    end

    coupling = grid_i != grid_j
    body = quote
        $(replaced[3]...)
        $iw_i, $iw_j = $weights_i′[$p], $weights_j′[$p]
        $gridindices_i, $gridindices_j = $_get_neighboringnodes($iw_i, $grid_i′, $iw_j, $grid_j′, Val($coupling))
        $ldofs_i, $ldofs_j = LinearIndices((size($gdofs_i, 1), size($gridindices_i)...)), LinearIndices((size($gdofs_j, 1), size($gridindices_j)...))
        $(lmat_init...)
        for $jp in eachindex($gridindices_j)
            $j = $gridindices_j[$jp]
            $(union(replaced[2], replaced[5])...)
            $J = vec(view($ldofs_j,:,$jp))
            for $ip in eachindex($gridindices_i)
                $i = $gridindices_i[$ip]
                $(union(replaced[1], replaced[4])...)
                $I = vec(view($ldofs_i,:,$ip))
                $(lmat_asm...)
            end
        end
        $dofs_i, $dofs_j = $_get_dofs($gdofs_i, $gridindices_i, $gdofs_j, $gridindices_j, Val($coupling))
        $(lmat2gmat...)
    end

    if !DEBUG
        body = :(@inbounds $body)
    end

    body = quote
        $check_arguments_for_P2G_Matrix($grid_i, $particles, $weights_i, $partition)
        $check_arguments_for_P2G_Matrix($grid_j, $particles, $weights_j, $partition)
        $(fillzeros...)
        $gdofs_i = LinearIndices((size($(gmats[1]),1)÷length($grid_i), size($grid_i)...))
        $gdofs_j = LinearIndices((size($(gmats[1]),2)÷length($grid_j), size($grid_j)...))
        @assert all(==((length($gdofs_i), length($gdofs_j))), map(size, ($(gmats...),)))
        $P2G((($grid_i′,$grid_j′), $particles, ($weights_i′,$weights_j′), $p) -> $body, $get_device($grid_i), Val($schedule), ($grid_i,$grid_j), $particles, ($weights_i,$weights_j), $partition)
    end

    esc(body)
end

@inline _get_neighboringnodes(iw_i, grid_i, iw_j, grid_j, ::Val{true}) = (@_propagate_inbounds_meta; (neighboringnodes(iw_i, grid_i), neighboringnodes(iw_j, grid_j)))
@inline _get_neighboringnodes(iw_i, grid_i, iw_j, grid_j, ::Val{false}) = (@_propagate_inbounds_meta; inds=neighboringnodes(iw_i, grid_i); (inds, inds))
@inline _get_dofs(gdofs_i, gridindices_i, gdofs_j, gridindices_j, ::Val{true}) = (@_propagate_inbounds_meta; (collect(vec(view(gdofs_i, :, gridindices_i))), collect(vec(view(gdofs_j, :, gridindices_j)))))
@inline _get_dofs(gdofs_i, gridindices_i, gdofs_j, gridindices_j, ::Val{false}) = (@_propagate_inbounds_meta; dofs=collect(vec(view(gdofs_j, :, gridindices_j))); (dofs, dofs))

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

@inline trySArray(x::Tensor) = SArray(x)
@inline trySArray(x::AbstractArray) = x
@inline trySArray(x::Real) = Scalar(x)

"""
    Tesserae.newton!(x::AbstractVector, f, ∇f,
                     maxiter = 100, atol = zero(eltype(x)), rtol = sqrt(eps(eltype(x))),
                     linsolve = (x,A,b) -> copyto!(x, A\\b), verbose = false)

A simple implementation of Newton's method.
The functions `f(x)` and `∇f(x)` should return the residual vector and its Jacobian, respectively.

```jldoctest
julia> function f(x)
           [(x[1]+3)*(x[2]^3-7)+18,
            sin(x[2]*exp(x[1])-1)]
       end
f (generic function with 1 method)

julia> function ∇f(x)
           u = exp(x[1])*cos(x[2]*exp(x[1])-1)
           [x[2]^3-7 3*x[2]^2*(x[1]+3)
            x[2]*u   u]
       end
∇f (generic function with 1 method)

julia> x = [0.1, 1.2];

julia> issuccess = Tesserae.newton!(x, f, ∇f)
true

julia> x ≈ [0,1]
true
```
"""
function newton!(
        x::AbstractVector, F, ∇F;
        maxiter::Int=100, atol::Real=zero(eltype(x)), rtol::Real=sqrt(eps(eltype(x))),
        linsolve=(x,A,b)->copyto!(x,A\b), backtracking::Bool=false, verbose::Bool=false)

    T = eltype(x)

    Fx = F(x)
    fx = f0 = norm(Fx)
    δx = similar(x)

    # previous step values
    x_prev, Fx_prev, fx_prev = similar(x), similar(Fx), fx

    iter = 0
    solved = f0 ≤ atol
    giveup = !isfinite(fx)

    if verbose
        newton_print_header(maxiter, atol, rtol)
        newton_print_row(maxiter, iter, fx, fx/f0)
    end

    while !(solved || giveup)
        @. x_prev = x
        @. Fx_prev = Fx
        fx_prev = fx

        linsolve(fillzero!(δx), ∇F(x), Fx)

        if backtracking
            ϕ0 = fx_prev * fx_prev / 2
            ϕ′0 = -fx_prev * fx_prev
            α, ok = newton_backtracking(one(T), ϕ0, ϕ′0) do α
                @. x = x_prev - α * δx # update `x`
                Fx = F(x)
                fx = norm(Fx)
                fx * fx / 2
            end
            ok || (giveup = true; break)
        else
            @. x = x_prev - δx
            Fx = F(x)
        end

        fx = norm(Fx)
        solved = fx ≤ max(atol, rtol*f0)
        giveup = ((iter += 1) ≥ maxiter || !isfinite(fx))

        verbose && newton_print_row(maxiter, iter, fx, fx/f0)
    end
    verbose && println()

    if giveup
        @. x = x_prev
        F(x)
        ∇F(x)
    end

    solved
end

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
    converged = false
    ϕα = ϕ(α)
    α_prev, ϕα_prev = α, ϕα
    for step in 1:maxiter
        ϕα ≤ ϕ0 + c*α*ϕ′0 && (converged = true; break)

        α_new = step == 1 ? quad_step(α, ϕα, ϕ0, ϕ′0, ρ_hi, ρ_lo) :
                            cubic_step(α, ϕα, α_prev, ϕα_prev, ϕ0, ϕ′0, ρ_hi, ρ_lo)
        α_new = clamp(α_new, α*ρ_lo, α*ρ_hi)
        α_prev, ϕα_prev = α, ϕα
        α = α_new
        ϕα = ϕ(α)

        abs(α) < eps(T)^T(2/3) && break
    end
    α, converged
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
        d ≥ 0 && return (-b + sqrt(d)) / 3a
    end
    ρ_lo * α
end
