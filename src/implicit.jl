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
struct DofMap{N, I <: AbstractVector{<: CartesianIndex}}
    masksize::Dims{N}
    indices::I # (dof, x, y, z)
end

DofMap(mask::AbstractArray{Bool}) = DofMap(size(mask), findall(mask))
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
    indices′ = maparray(I->CartesianIndex(1,Base.tail(Tuple(I))...), dofmap.indices)
    @boundscheck checkbounds(A′, indices′)
    @inbounds view(A′, indices′)
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
9×9 view(::SparseArrays.SparseMatrixCSC{Float64, Int64}, [1, 2, 3, 12, 13, 14, 23, 24, 25], [1, 2, 3, 12, 13, 14, 23, 24, 25]) with eltype Float64:
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
function create_sparse_matrix(it::Interpolation, mesh::AbstractMesh; ndofs::Int = ndims(mesh))
    create_sparse_matrix(Float64, it, mesh; ndofs)
end
function create_sparse_matrix(::Type{T}, it::Interpolation, mesh::CartesianMesh{dim}; ndofs::Int = dim) where {T, dim}
    dims = size(mesh)
    spy = falses(ndofs, prod(dims), ndofs, prod(dims))
    LI = LinearIndices(dims)
    mesh = CartesianMesh(float(T), 1, map(d->(0,d-1), dims)...)
    for i in eachindex(mesh)
        unit = (gridspan(it) - 1) * oneunit(i)
        indices = intersect((i-unit):(i+unit), eachindex(mesh))
        for j in indices
            spy[1:ndofs, LI[i], 1:ndofs, LI[j]] .= true
        end
    end
    create_sparse_matrix(T, reshape(spy, ndofs*prod(dims), ndofs*prod(dims)))
end

function create_sparse_matrix(::Type{T}, spy::AbstractMatrix{Bool}) where {T}
    I = findall(vec(spy))
    V = zeros(T, length(I))
    SparseArrays.sparse_sortedlinearindices!(I, V, size(spy)...)
end

function create_sparse_matrix(::Type{T}, meshes::Tuple{Vararg{UnstructuredMesh, N}}; ndofs::NTuple{N, Int}) where {T, N}
    gdofs = generate_dofs(meshes, ndofs)
    ttldofs = sum(length, gdofs)
    spy = fill(false, ttldofs, ttldofs)
    @threaded for i in 1:N
        mesh_i = meshes[i]
        gdofs_i = gdofs[i]
        for c_i in 1:ncells(mesh_i)
            cellnodes_i = cellnodeindices(mesh_i, c_i)
            primarycellnodes_i = cellnodes_i[primarynodes_indices(cellshape(mesh_i))]
            celldofs_i = gdofs_i[:, cellnodes_i]
            for j in 1:N
                if i == j
                    spy[celldofs_i, celldofs_i] .= true
                else # check shared cells
                    mesh_j = meshes[j]
                    gdofs_j = gdofs[j]
                    for c_j in 1:ncells(mesh_j)
                        cellnodes_j = cellnodeindices(mesh_j, c_j)
                        primarycellnodes_j = cellnodes_j[primarynodes_indices(cellshape(mesh_j))]
                        if mesh_i[primarycellnodes_i] ≈ mesh_j[primarycellnodes_j] # TODO: better way because this is very slow
                            celldofs_j = gdofs_j[:, cellnodes_j]
                            spy[celldofs_i, celldofs_j] .= true
                        end
                    end
                end
            end
        end
    end
    create_sparse_matrix(T, spy)
end
create_sparse_matrix(meshes::Tuple{Vararg{UnstructuredMesh}}; ndofs::Dims) = create_sparse_matrix(Float64, meshes; ndofs)

function generate_dofs(meshes::Tuple{Vararg{UnstructuredMesh, N}}, ndofs::Tuple{Vararg{Int, N}}) where {N}
    @assert all(i -> ndofs[i] != 0 || meshes[i] ≈ meshes[i-1], 1:N)
    ttldofs = map(i -> ndofs[i]*length(meshes[i]), 1:N)
    ntuple(Val(N)) do i
        offset = _get_valid_offset(ttldofs, i)
        LinearIndices((_get_valid_ndofs(ndofs, i), length(meshes[i]))) .+ offset
    end
end
_get_valid_ndofs(ndofs, i) = ndofs[i]==0 ? _get_valid_ndofs(ndofs, i-1) : ndofs[i]
_get_valid_offset(ttldofs, i) = ttldofs[i]==0 ? _get_valid_offset(ttldofs, i-1) : sum(ttldofs[1:i-1])

create_sparse_matrix(::Type{T}, mesh::UnstructuredMesh{<: Any, dim}; ndofs::Int = dim) where {T, dim} = create_sparse_matrix(T, (mesh,); ndofs=(ndofs,))
create_sparse_matrix(mesh::UnstructuredMesh{<: Any, dim}; ndofs::Int = dim) where {dim} = create_sparse_matrix(Float64, mesh; ndofs)

"""
    extract(matrix::AbstractMatrix, dofmap_row::DofMap, dofmap_col::DofMap = dofmap_row)

Extract the active degrees of freedom of a matrix.
"""
function extract(S::AbstractMatrix, dofmap_i::DofMap, dofmap_j::DofMap = dofmap_i)
    m, n = prod(dofmap_i.masksize), prod(dofmap_j.masksize)
    @assert size(S) == (m, n)
    I = view(LinearIndices(dofmap_i.masksize), dofmap_i.indices)
    J = view(LinearIndices(dofmap_j.masksize), dofmap_j.indices)
    view(S, I, J)
end

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
    @P2G_Matrix grid=>(i,j) particles=>p mpvalues=>(ip,jp) [space] begin
        equations...
    end

Particle-to-grid transfer macro for assembling a global matrix.
A typical global stiffness matrix can be assembled as follows:

```julia
@P2G_Matrix grid=>(i,j) particles=>p mpvalues=>(ip,jp) begin
    K[i,j] = @∑ ∇w[ip] ⋅ c[p] ⋅ ∇w[jp] * V[p]
end
```

where `c` and `V` denote the stiffness (symmetric fourth-order) tensor and the volume, respectively.
It is recommended to create global stiffness `K` using [`create_sparse_matrix`](@ref).
"""
macro P2G_Matrix(grid_pair, particles_pair, mpvalues_pair, equations)
    P2G_Matrix_macro(QuoteNode(:nothing), grid_pair, particles_pair, mpvalues_pair, nothing, equations)
end
macro P2G_Matrix(grid_pair, particles_pair, mpvalues_pair, space, equations)
    P2G_Matrix_macro(QuoteNode(:nothing), grid_pair, particles_pair, mpvalues_pair, space, equations)
end
macro P2G_Matrix(schedule::QuoteNode, grid_pair, particles_pair, mpvalues_pair, equations)
    P2G_Matrix_macro(schedule, grid_pair, particles_pair, mpvalues_pair, nothing, equations)
end
macro P2G_Matrix(schedule::QuoteNode, grid_pair, particles_pair, mpvalues_pair, space, equations)
    P2G_Matrix_macro(schedule, grid_pair, particles_pair, mpvalues_pair, space, equations)
end

function P2G_Matrix_macro(schedule::QuoteNode, grid_pair, particles_pair, mpvalues_pair, space, equations)
    grid, (i,j) = unpair2(grid_pair)
    particles, p = unpair(particles_pair)
    mpvalues, (ip,jp) = unpair2(mpvalues_pair)
    @gensym mp gridindices localdofs I J dofs fulldofs

    @assert equations.head == :block
    Base.remove_linenums!(equations)
    replace_dollar_by_identity!(equations)
    sumornot = map(ex->_issumexpr(ex.args[2]), equations.args)

    sum_equations = equations.args[sumornot]
    nosum_equations = equations.args[.!sumornot]

    @assert all(map(ex->issumexpr(ex, i, j), sum_equations))

    pairs = [grid=>i, grid=>j, particles=>p, mp=>ip, mp=>jp]
    vars = [Set{Expr}(), Set{Expr}(), Set{Expr}(), Set{Expr}(), Set{Expr}()]

    init_global_matrices = Expr[]
    global_matrices = Symbol[]
    create_local_matrices = Expr[]
    assemble_local_matrices = Expr[]
    add_local_to_global = Expr[]
    for ex in sum_equations
        rhs = remove_∑(ex.args[2])
        complete_sumeq_rhs_expr!(Meta.quot(rhs), pairs, vars) # rhs
        lhs = ex.args[1]
        @assert Meta.isexpr(lhs, :ref)
        globalname = lhs.args[1]
        localname = gensym(globalname)

        Meta.isexpr(ex, :(=)) && push!(init_global_matrices, :(Tesserae.fillzero!($globalname)))
        push!(global_matrices, globalname)
        push!(create_local_matrices, :($localname = Array{eltype($globalname)}(undef, length($localdofs), length($localdofs))))
        push!(assemble_local_matrices, :(@inbounds $localname[$I,$J] .= $trySArray($rhs))) # converting `Tensor` to `SArray` is faster for setindex!
        push!(add_local_to_global, :(Tesserae.add!($globalname, $dofs, $dofs, $localname)))
    end

    grid_sums = Expr[]
    for ex in nosum_equations
        lhs = ex.args[1]
        @assert ex.head == :(+=)
        @assert Meta.isexpr(lhs, :ref) && all(lhs.args[2:end] .== i) # currently support only `A[i,i] = ...`
        complete_parent_from_index!(ex.args[2], [grid=>i])
        globalname = lhs.args[1]
        push!(grid_sums, :(Tesserae.add!($globalname, $I, $I, $(ex.args[2]))))
    end

    body = quote
        $(vars[3]...)
        $mp = $mpvalues[$p]
        $gridindices = neighboringnodes($mp, $grid)
        $localdofs = LinearIndices((size($fulldofs, 1), size($gridindices)...))
        $(create_local_matrices...)
        for $jp in eachindex($gridindices)
            $j = $gridindices[$jp]
            $(union(vars[2], vars[5])...)
            $J = vec(view($localdofs,:,$jp))
            for $ip in eachindex($gridindices)
                $i = $gridindices[$ip]
                $(union(vars[1], vars[4])...)
                $I = vec(view($localdofs,:,$ip))
                $(assemble_local_matrices...)
            end
        end
        $dofs = collect(vec(view($fulldofs, :, $gridindices)))
        $(add_local_to_global...)
    end

    if !DEBUG
        body = :(@inbounds $body)
    end

    body = quote
        $check_arguments_for_P2G_Matrix($grid, $particles, $mpvalues, $space)
        $(init_global_matrices...)
        $fulldofs = LinearIndices((size($(first(global_matrices)),1)÷length($grid), size($grid)...))
        @assert all(==((length($fulldofs), length($fulldofs))), map(size, ($(global_matrices...),)))
        $P2G(($grid, $particles, $mpvalues, $p) -> $body, $get_device($grid), Val($schedule), $grid, $particles, $mpvalues, $space)
    end

    if !isempty(grid_sums)
        body = quote
            $body
            for $i in eachindex($grid)
                $I = vec(view($fulldofs,:,$i))
                $(grid_sums...)
            end
        end
    end

    esc(body)
end

function check_arguments_for_P2G_Matrix(grid, particles, mpvalues, space)
    check_arguments_for_P2G(grid, particles, mpvalues, space)
    @assert get_device(grid) isa CPUDevice
end

function unpair2(expr::Expr)
    @assert expr.head==:call && expr.args[1]==:(=>) && isa(expr.args[2],Symbol) && Meta.isexpr(expr.args[3],:tuple) && length(expr.args[3].args)==2
    expr.args[2], (expr.args[3].args...,)
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

    compact(val, pad) = rpad(sprint(show, val; context = :compact=>true), pad)
    T = eltype(x)

    Fx = F(x)
    fx = f0 = norm(Fx)
    δx = similar(x)

    if backtracking
        # previous step values
        x′, Fx′, fx′ = copy(x), copy(Fx), fx
    end

    iter = 0
    solved = f0 ≤ atol
    giveup = !isfinite(fx)

    if verbose
        println(" # ≤ ", compact(maxiter,4), "   ‖f‖ ≤ ", compact(atol,11), "   ‖f‖/‖f₀‖ ≤ ", compact(rtol,11))
        println("---------  ------------------  -----------------------")
        println(" ", compact(iter,7), "    ", compact(fx,16), "    ", compact(fx/f0,16))
    end

    while !(solved || giveup)
        linsolve(fillzero!(δx), ∇F(x), Fx)

        @. x -= δx
        Fx = F(x)
        fx = norm(Fx)

        if backtracking
            α = one(T)
            while fx ≥ fx′
                α = α^2 * fx′ / 2(fx + α*fx′ - fx′)
                @. x = x′ - α * δx
                Fx = F(x)
                fx = norm(Fx)
            end
            @. x′ = x
            @. Fx′ = Fx
            fx′ = fx
        end

        solved = fx ≤ max(atol, rtol*f0)
        giveup = ((iter += 1) ≥ maxiter || !isfinite(fx))

        if verbose
            println(" ", compact(iter,7), "    ", compact(fx,16), "    ", compact(fx/f0,16))
        end
    end

    verbose && println()
    solved
end
