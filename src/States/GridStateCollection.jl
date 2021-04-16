struct GridStateCollection{T} <: AbstractCollection{2}
    data::T
    dofindices::PointToDofIndices
end

GridStateCollection(x::UnionGridState) = GridStateCollection(nonzeros(x), dofindices(x))

Base.length(x::GridStateCollection) = length(x.dofindices) # == npoints
Base.getindex(x::GridStateCollection, i::Int) = (@_propagate_inbounds_meta; view(Collection{1}(x.data), x.dofindices[i]))

# for ∑ᵢ(vᵢ * N) and ∑ᵢ(vᵢ ⊗ ∇(N))
for op in (:(Base.:*), :(Tensorial.:⊗))
    @eval begin
        $op(x::UnionGridState, y::AbstractCollection{2}) = $op(GridStateCollection(x), y)
        $op(x::AbstractCollection{2}, y::UnionGridState) = $op(x, GridStateCollection(y))
    end
end
