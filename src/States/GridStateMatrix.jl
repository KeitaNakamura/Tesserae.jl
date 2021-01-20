struct GridStateMatrix{T, ElType}
    A::SparseMatrixCOO{ElType}
    dofindices::Vector{Vector{Int}}
end

# stiffness matrix
function gridstate_matrix(::Type{Vec{dim, T}}, dofindices::Vector{Vector{Int}}) where {dim, T}
    GridStateMatrix{Vec{dim, T}, T}(SparseMatrixCOO{T}(), dofindices)
end

Base.empty!(x::GridStateMatrix) = empty!(x.A)
Base.push!(x::GridStateMatrix, args...) = push!(x.A, args...)

sparse(x::GridStateMatrix) = sparse(x.A)
