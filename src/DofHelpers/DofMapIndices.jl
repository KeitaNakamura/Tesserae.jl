struct DofMapIndices{dim} <: AbstractArray{Union{Nothing, Int}, dim}
    indices::Array{Int, dim}
end

indices(dofmap::DofMap) = DofMapIndices(dofmap.indices)

Base.parent(I::DofMapIndices) = DofMap(I.indices)
Base.IndexStyle(::Type{<: DofMapIndices}) = IndexLinear()
Base.size(I::DofMapIndices) = size(I.indices)
Base.getindex(I::DofMapIndices, i::Int) = (@_propagate_inbounds_meta; j = I.indices[i]; ifelse(j == -1, nothing, j))

ndofs(I::DofMapIndices; dof = 1) = ndofs(parent(I); dof)
