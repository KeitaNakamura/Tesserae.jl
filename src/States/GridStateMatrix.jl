struct GridStateMatrix{T, ElType}
    A::SparseMatrixCOO{ElType}
    dofindices::Vector{Vector{Int}}
end

# for vector field
function gridstate_matrix(::Type{<: Tensor{Tuple{dim, dim}, T}}, dofindices::Vector{Vector{Int}}) where {dim, T}
    GridStateMatrix{Tensor{Tuple{dim,dim}}, T}(SparseMatrixCOO{T}(), dofindices)
end

# for scalar field
function gridstate_matrix(::Type{T}, dofindices::Vector{Vector{Int}}) where {T <: Real}
    GridStateMatrix{T, T}(SparseMatrixCOO{T}(), dofindices)
end

Base.empty!(x::GridStateMatrix) = empty!(x.A)
Base.push!(x::GridStateMatrix, args...) = push!(x.A, args...)

sparse(x::GridStateMatrix) = sparse(x.A)
