struct DofMapIndices{dim} <: AbstractArray{Union{Nothing, Int}, dim}
    dofmap::DofMap{dim}
end

indices(dofmap::DofMap) = DofMapIndices(dofmap)

Base.parent(I::DofMapIndices) = I.dofmap
Base.IndexStyle(::Type{<: DofMapIndices}) = IndexLinear()
Base.size(I::DofMapIndices) = size(I.dofmap)
Base.getindex(I::DofMapIndices, i::Int) = (@_propagate_inbounds_meta; j = I.dofmap.indices[i]; ifelse(j == -1, nothing, j))

ndofs(I::DofMapIndices; dof = 1) = ndofs(parent(I); dof)
