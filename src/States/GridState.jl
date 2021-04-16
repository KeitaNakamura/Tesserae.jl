struct GridState{dim, T, V} <: AbstractArray{T, dim}
    sp::SparseArray{T, dim, V}
    dofindices::PointToDofIndices
end

function gridstate(::Type{T}, dofmap::DofMap, dofindices::PointToDofIndices) where {T}
    GridState(SparseArray{T}(dofmap), dofindices)
end

Base.size(A::GridState) = size(A.sp)
nonzeros(A::GridState) = nonzeros(A.sp)
nzindices(A::GridState) = nzindices(A.sp)
dofindices(A::GridState) = A.dofindices
nnz(A::GridState) = nnz(A.sp)

totalnorm(A::GridState) = norm(flatview(nonzeros(A.sp)))
Base.zero(A::GridState) = GridState(zero(A.sp), dofindices(A))

@inline function Base.getindex(A::GridState{dim, T}, I::Vararg{Int, dim}) where {dim, T}
    @boundscheck checkbounds(A, I...)
    @inbounds A.sp[I...]
end

@inline function Base.setindex!(A::GridState{dim}, v, I::Vararg{Int, dim}) where {dim}
    @boundscheck checkbounds(A, I...)
    @inbounds A.sp[I...] = v
    A
end

zeros!(v::AbstractVector{T}, n) where {T} = (resize!(v, n); fill!(v, zero(T)); v)
zeros!(v) = (fill!(v, zero(eltype(v))); v)
zeros!(A::GridState) = (zeros!(nonzeros(A), nnz(A)); A)
Base.resize!(A::GridState) = (resize!(nonzeros(A), nnz(A)); A)


struct GridStateThreads{dim, T, V} <: AbstractArray{T, dim}
    state::GridState{dim, T, V}
    ptranges::Vector{UnitRange{Int}}
    state_threads::Vector{GridState{dim, T, V}}
end

Base.size(A::GridStateThreads) = size(A.state)
nonzeros(A::GridStateThreads) = nonzeros(A.state)
nzindices(A::GridStateThreads) = nzindices(A.state)
dofindices(A::GridStateThreads) = dofindices(A.state)
nnz(A::GridStateThreads) = nnz(A.state)

totalnorm(A::GridStateThreads) = totalnorm(A.state)
Base.zero(A::GridStateThreads) = zero(A.state)

@inline function Base.getindex(A::GridStateThreads{dim, T}, I::Vararg{Int, dim}) where {dim, T}
    @boundscheck checkbounds(A, I...)
    @inbounds A.state[I...]
end

@inline function Base.setindex!(A::GridStateThreads{dim}, v, I::Vararg{Int, dim}) where {dim}
    @boundscheck checkbounds(A, I...)
    @inbounds A.state[I...] = v
    A
end

function zeros!(A::GridStateThreads)
    zeros!(A.state)
    foreach(zeros!, A.state_threads)
    A
end

function Base.resize!(A::GridStateThreads)
    resize!(A.state)
    foreach(resize!, A.state_threads)
    A
end
