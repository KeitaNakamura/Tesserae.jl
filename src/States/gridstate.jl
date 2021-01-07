struct GridState{dim, T} <: AbstractArray{T, dim}
    nzval::Vector{T}
    indices::DofMapIndices{dim}
    dofindices::Vector{Vector{Int}}
end

function gridstate(::Type{T}, dofmap::DofMap, dofindices::Vector{Vector{Int}}) where {T}
    nzval = zeros(T, ndofs(dofmap))
    GridState(nzval, indices(dofmap), dofindices)
end

Base.size(A::GridState) = size(indices(A))
indices(A::GridState) = A.indices

@inline function Base.getindex(A::GridState{dim, T}, I::Vararg{Int, dim}) where {dim, T}
    @boundscheck checkbounds(A, I...)
    @inbounds begin
        index = indices(A)[I...]
        index === nothing ? zero(T) : A.nzval[index]
    end
end

@inline function Base.setindex!(A::GridState{dim}, v, I::Vararg{Int, dim}) where {dim}
    @boundscheck checkbounds(A, I...)
    @inbounds begin
        index = indices(A)[I...]
        index === nothing ? throw(ArgumentError("GridState: setindex! at invalid index")) : A.nzval[index] = v
    end
    A
end

nonzeros(S::GridState) = S.nzval
nnz(S::GridState) = ndofs(indices(S))

zeros!(v::AbstractVector{T}, n) where {T} = (resize!(v, n); fill!(v, zero(T)); v)
zeros!(v) = (fill!(v, zero(eltype(v))); v)
zeros!(S::GridState) = (zeros!(nonzeros(S), nnz(S)); S)
