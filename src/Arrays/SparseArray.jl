struct SparseArray{T, dim, V <: AbstractVector{T}} <: AbstractArray{T, dim}
    nzval::V
    indices::DofMapIndices{dim}
end

function SparseArray(nzval::AbstractVector, dofmap::DofMap)
    SparseArray(nzval, indices(dofmap))
end

function SparseArray{T}(dofmap::DofMap) where {T}
    nzval = zeros(T, ndofs(dofmap))
    SparseArray(nzval, dofmap)
end

nonzeros(S::SparseArray) = S.nzval
nzindices(A::SparseArray) = A.indices

Base.size(A::SparseArray) = size(nzindices(A))
nnz(S::SparseArray) = ndofs(nzindices(S))

Base.zero(S::SparseArray{T}) where {T} = SparseArray{T}(parent(nzindices(S)))

@inline function Base.getindex(A::SparseArray{T, dim}, I::Vararg{Int, dim}) where {T, dim}
    @boundscheck begin
        checkbounds(A, I...)
        @assert length(nonzeros(A)) == nnz(A)
    end
    @inbounds begin
        index = nzindices(A)[I...]
        index === nothing ? zero(T) : nonzeros(A)[index]
    end
end

@inline function Base.setindex!(A::SparseArray{<: Any, dim}, v, I::Vararg{Int, dim}) where {dim}
    @boundscheck begin
        checkbounds(A, I...)
        @assert length(nonzeros(A)) == nnz(A)
    end
    @inbounds begin
        index = nzindices(A)[I...]
        index === nothing ? throw(ArgumentError("SparseArray: setindex! at invalid index")) :
                            nonzeros(A)[index] = v
    end
    A
end
