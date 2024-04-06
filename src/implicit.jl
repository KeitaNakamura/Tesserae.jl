struct DofMap{dim, N, I <: AbstractVector{CartesianIndex{N}}}
    dimension::Int
    gridsize::Dims{dim}
    indices::I # (direction, x, y, z)
end

DofMap(mask::AbstractArray{Bool}) = DofMap(size(mask, 1), Base.tail(size(mask)), findall(mask))
DofMap(dims::Dims) = DofMap(dims[1], Base.tail(dims), vec(CartesianIndices(dims)))

ndofs(dofmap::DofMap) = length(dofmap.indices)

function (dofmap::DofMap{dim, N})(A::AbstractArray{T, dim}) where {dim, N, T <: Vec}
    @assert dim+1 == N
    A′ = reinterpret(reshape, eltype(T), A)
    @boundscheck checkbounds(A′, dofmap.indices)
    @inbounds view(A′, dofmap.indices)
end

function (dofmap::DofMap{dim, N})(A::AbstractArray{T, dim}) where {dim, N, T <: Real}
    @assert dim+1 == N
    A′ = reshape(A, 1, size(A)...)
    indices′ = maparray(I->CartesianIndex(1,Base.tail(Tuple(I))...), dofmap.indices)
    @boundscheck checkbounds(A′, indices′)
    @inbounds view(A′, indices′)
end

function create_sparse_matrix(::Type{Vec{dim}}, it::Interpolation, mesh::AbstractMesh) where {dim}
    create_sparse_matrix(Vec{dim, Float64}, it, mesh)
end
function create_sparse_matrix(::Type{Vec{dim, T}}, it::Interpolation, mesh::CartesianMesh) where {dim, T}
    dims = size(mesh)
    spy = falses(dim, prod(dims), dim, prod(dims))
    LI = LinearIndices(dims)
    mesh = CartesianMesh(float(T), 1, map(d->(0,d-1), dims)...)
    for i in eachindex(mesh)
        unit = gridspan(it) * oneunit(i)
        indices = intersect((i-unit):(i+unit), eachindex(mesh))
        for j in indices
            spy[1:dim, LI[i], 1:dim, LI[j]] .= true
        end
    end
    create_sparse_matrix(T, reshape(spy, dim*prod(dims), dim*prod(dims)))
end

function create_sparse_matrix(::Type{T}, spy::AbstractMatrix{Bool}) where {T}
    I = findall(vec(spy))
    V = Vector{T}(undef, length(I))
    SparseArrays.sparse_sortedlinearindices!(I, V, size(spy)...)
end

function getsubset(S::AbstractMatrix, dofmap::DofMap)
    n = dofmap.dimension * prod(dofmap.gridsize)
    @assert size(S) == (n, n)
    I = view(LinearIndices((dofmap.dimension, dofmap.gridsize...)), dofmap.indices)
    S[I, I]
end

function add!(A::SparseMatrixCSC, I::AbstractVector{Int}, J::AbstractVector{Int}, K::AbstractMatrix)
    @boundscheck checkbounds(A, I, J)
    @assert issorted(I)
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
            if I[i] == row
                vals[k] += K[i,j]
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

macro P2G_Matrix(grid_pair, particles_pair, mpvalues_pair, equations)
    P2G_Matrix_macro(schedule, grid_pair, particles_pair, mpvalues_pair, nothing, equations)
end
macro P2G_Matrix(grid_pair, particles_pair, mpvalues_pair, blockspace, equations)
    P2G_Matrix_macro(schedule, grid_pair, particles_pair, mpvalues_pair, blockspace, equations)
end
macro P2G_Matrix(schedule::QuoteNode, grid_pair, particles_pair, mpvalues_pair, equations)
    P2G_Matrix_macro(schedule, grid_pair, particles_pair, mpvalues_pair, nothing, equations)
end
macro P2G_Matrix(schedule::QuoteNode, grid_pair, particles_pair, mpvalues_pair, blockspace, equations)
    P2G_Matrix_macro(schedule, grid_pair, particles_pair, mpvalues_pair, blockspace, equations)
end

function P2G_Matrix_macro(schedule::QuoteNode, grid_pair, particles_pair, mpvalues_pair, blockspace, equations)
    grid, (i,j) = unpair2(grid_pair)
    particles, p = unpair(particles_pair)
    mpvalues, (ip,jp) = unpair2(mpvalues_pair)
    @gensym mp gridindices localdofs I J dofs fulldofs

    @assert equations.head == :block
    Base.remove_linenums!(equations)
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
    assertions = Expr[]
    for ex in sum_equations
        rhs = remove_∑(ex.args[2])
        complete_sumeq_rhs_expr!(Meta.quot(rhs), pairs, vars) # rhs
        lhs = ex.args[1]
        @assert Meta.isexpr(lhs, :ref)
        globalname = lhs.args[1]
        localname = gensym(globalname)

        Meta.isexpr(ex, :(=)) && push!(init_global_matrices, :(Sequoia.fillzero!($globalname)))
        push!(global_matrices, globalname)
        push!(create_local_matrices, :($localname = Array{eltype($globalname)}(undef, length($localdofs), length($localdofs))))
        push!(assemble_local_matrices, :(@inbounds $localname[$I,$J] .= $trySArray($rhs))) # converting `Tensor` to `SArray` is faster for setindex!
        push!(add_local_to_global, :(Sequoia.add!($globalname, $dofs, $dofs, $localname)))
        push!(assertions, :(@assert $globalname isa AbstractMatrix && size($globalname, 1) == size($globalname, 2)))
    end
    push!(assertions, :(@assert allequal($([:(size($m)) for m in global_matrices]...,))))

    grid_sums = Expr[]
    for ex in nosum_equations
        lhs = ex.args[1]
        @assert ex.head == :(+=)
        @assert Meta.isexpr(lhs, :ref) && all(lhs.args[2:end] .== i) # currently support only `A[i,i] = ...`
        complete_parent_from_index!(ex.args[2], [grid=>i])
        globalname = lhs.args[1]
        push!(grid_sums, :(Sequoia.add!($globalname, $I, $I, $(ex.args[2]))))
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
                $(union(vars[1], vars[4])...)
                $i = $gridindices[$ip]
                $I = vec(view($localdofs,:,$ip))
                $(assemble_local_matrices...)
            end
        end
        $dofs = collect(vec(view($fulldofs, :, $gridindices)))
        $(add_local_to_global...)
    end

    body = quote
        $P2G(Val($schedule), $grid, $particles, $mpvalues, $blockspace) do $p, $grid, $particles, $mpvalues
            Base.@_inline_meta
            $body
        end
    end

    body = quote
        $(assertions...)
        $(init_global_matrices...)
        $fulldofs = LinearIndices((size($(first(global_matrices)),1)÷length($grid), size($grid)...))
        $body
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

function unpair2(expr::Expr)
    @assert expr.head==:call && expr.args[1]==:(=>) && isa(expr.args[2],Symbol) && Meta.isexpr(expr.args[3],:tuple) && length(expr.args[3].args)==2
    expr.args[2], (expr.args[3].args...,)
end

@inline trySArray(x::Tensor) = SArray(x)
@inline trySArray(x::AbstractArray) = x
