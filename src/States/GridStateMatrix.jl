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

Base.empty!(x::GridStateMatrix) = empty!(x.A)
Base.push!(x::GridStateMatrix, args...) = push!(x.A, args...)

sparse(x::GridStateMatrix) = sparse(x.A)
freedofs(x::GridStateMatrix) = x.freedofs


function Base.:\(A::GridStateMatrix, b::GridState)
    # TODO: This doesn't work well because freedofs is created for Vector field
    @assert A.dofindices === b.dofindices
    dofs = freedofs(A)
    x = copyzero(b)
    AA = sparse(A)
    bb = flatview(nonzeros(b))
    xx = flatview(nonzeros(x))
    xx[dofs] = AA[dofs, dofs] \ bb[dofs]
    x
end
