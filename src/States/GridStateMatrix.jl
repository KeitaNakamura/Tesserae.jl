struct GridStateMatrix{T, ElType}
    A::SparseMatrixCOO{ElType}
    dofindices::Vector{Vector{Int}}
    freedofs::Vector{Int}
end

# for vector field
function gridstate_matrix(::Type{<: Tensor{Tuple{dim, dim}, T}}, dofindices::Vector{Vector{Int}}, freedofs::Vector{Int}) where {dim, T}
    GridStateMatrix{Tensor{Tuple{dim,dim}}, T}(SparseMatrixCOO{T}(), dofindices, freedofs)
end

# for scalar field
function gridstate_matrix(::Type{T}, dofindices::Vector{Vector{Int}}, freedofs::Vector{Int}) where {T <: Real}
    GridStateMatrix{T, T}(SparseMatrixCOO{T}(), dofindices, freedofs)
end

sparse(x::GridStateMatrix) = sparse(x.A)

function reinit!(x::GridStateMatrix{Tensor{Tuple{dim, dim}}, T}) where {dim, T}
    empty!(x.A)
    @inbounds for dofs in x.dofindices
        n = length(dofs)
        for index in CartesianIndices((n, n))
            i, j = Tuple(index)
            I = _compute_range(dofs[i], dim) # dof range
            J = _compute_range(dofs[j], dim) # dof range
            # if eltype of mat is scalar, create ScalarMatrix
            push!(x.A, Zeros{T}(dim, dim), I, J)
        end
    end
    sparse!(x.A)
    x
end

function zeros!(x::GridStateMatrix)
    nzval = nonzeros(sparse(x))
    zeros!(nzval)
    x
end

function add!(x::GridStateMatrix, mat, I::AbstractVector{Int}, J::AbstractVector{Int})
    A = sparse(x)
    @boundscheck checkbounds(A, I, J)
    rows = rowvals(A)
    vals = nonzeros(A)
    @inbounds for (je, j) in enumerate(J)
        ie = 1
        for index in nzrange(A, j)
            if I[ie] == rows[index]
                vals[index] += mat[ie, je]
                ie += 1
                ie > length(I) && break
            end
        end
        ie ≤ length(I) && error("given indices are not found")
    end
end
function add!(x::GridStateMatrix, vec, I::AbstractVector{Int})
    A = sparse(x)
    @boundscheck checkbounds(A, I, I)
    rows = rowvals(A)
    vals = nonzeros(A)
    @inbounds for (je, j) in enumerate(I)
        ie = je
        for index in nzrange(A, j)
            if I[ie] == rows[index]
                vals[index] += vec[ie]
                ie += 1 # just for detecting error
                break
            end
        end
        ie == je && error("given indices are not found")
    end
end

freedofs(x::GridStateMatrix) = x.freedofs

function solve!(A::GridStateMatrix, b::GridState)
    # TODO: This doesn't work well because freedofs is created for Vector field
    @assert A.dofindices === b.dofindices
    dofs = freedofs(A)
    x = copyzero(b)
    AA = sparse(A)
    bb = flatview(nonzeros(b))
    xx = flatview(nonzeros(x))
    xx[dofs] = AA[dofs, dofs] \ bb[dofs]
    # fill bb with zero for dirichlet boundary conditions
    # fixeddofs = setdiff(eachindex(bb), dofs)
    # bb[fixeddofs] .= zero(eltype(bb))
    x
end
