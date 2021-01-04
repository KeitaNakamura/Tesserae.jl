"""
    SparseArray([T = Float64,] dofmap)

# Examples
```jldoctest
julia> dofmap = DofMap(5, 5);

julia> dofmap[1:2, 2:3] .= true; dofmap
5×5 DofMap{2}:
 0  1  1  0  0
 0  1  1  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0

julia> count!(dofmap)
4

julia> S = SparseArray(dofmap)
5×5 SparseArray{2,Float64}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> S[1,2] = 1; S
5×5 SparseArray{2,Float64}:
 0.0  1.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> S[1,1] = 1
ERROR: ArgumentError: SparseArray: setindex! at invalid index
```
"""
struct SparseArray{dim, T} <: AbstractArray{T, dim}
    nzval::Vector{T}
    indices::DofMapIndices{dim}
end

function SparseArray(nzval::Vector, dofmap::DofMap)
    SparseArray(nzval, indices(dofmap))
end

function SparseArray(::Type{T}, dofmap::DofMap) where {T}
    nzval = zeros(T, ndofs(dofmap))
    SparseArray(nzval, dofmap)
end

SparseArray(dofmap::DofMap) = SparseArray(Float64, dofmap)

Base.size(A::SparseArray) = size(indices(A))
indices(A::SparseArray) = A.indices

@inline function Base.getindex(A::SparseArray{dim, T}, I::Vararg{Int, dim}) where {dim, T}
    @boundscheck checkbounds(A, I...)
    @inbounds begin
        index = indices(A)[I...]
        index === nothing ? zero(T) : A.nzval[index]
    end
end

@inline function Base.setindex!(A::SparseArray{dim}, v, I::Vararg{Int, dim}) where {dim}
    @boundscheck checkbounds(A, I...)
    @inbounds begin
        index = indices(A)[I...]
        index === nothing ? throw(ArgumentError("SparseArray: setindex! at invalid index")) : A.nzval[index] = v
    end
    A
end

nonzeros(S::SparseArray) = S.nzval
nnz(S::SparseArray) = ndofs(indices(S))

zeros!(v::AbstractVector{T}, n) where {T} = (resize!(v, n); fill!(v, zero(T)); v)
zeros!(v) = (fill!(v, zero(eltype(v))); v)
zeros!(S::SparseArray) = (zeros!(nonzeros(S), nnz(S)); S)


"""
    SparseMatrixCOO([N = 0])

Construct sparse matrix using COOrdinate format.

# Examples
```jldoctest
julia> S = SparseMatrixCOO();

julia> push!(S, [1.0 2.0; 3.0 4.0], [1, 2]);

julia> sparse(S)
2×2 SparseArrays.SparseMatrixCSC{Float64,Int64} with 4 stored entries:
  [1, 1]  =  1.0
  [2, 1]  =  3.0
  [1, 2]  =  2.0
  [2, 2]  =  4.0

julia> push!(S, [1.0 1.0; 1.0 1.0], [1, 2], [2, 3]);

julia> sparse(S, 4, 4)
4×4 SparseArrays.SparseMatrixCSC{Float64,Int64} with 6 stored entries:
  [1, 1]  =  1.0
  [2, 1]  =  3.0
  [1, 2]  =  3.0
  [2, 2]  =  5.0
  [1, 3]  =  1.0
  [2, 3]  =  1.0
```
"""
struct SparseMatrixCOO{T}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{T}
end

function SparseMatrixCOO(N::Int = 0)
    I = Int[]
    J = Int[]
    V = Float64[]
    sizehint!(I, N)
    sizehint!(J, N)
    sizehint!(V, N)
    SparseMatrixCOO(I, J, V)
end

function Base.push!(S::SparseMatrixCOO, s::AbstractMatrix, I::AbstractVector{Int}, J::AbstractVector{Int})
    m = length(I)
    n = length(J)
    append!(S.V, s)
    @inbounds for j in 1:n
        append!(S.I, I)
        for i in 1:m
            push!(S.J, J[j])
        end
    end
    S
end

function Base.push!(S::SparseMatrixCOO, s::AbstractMatrix, dofs::AbstractVector{Int})
    push!(S, s, dofs, dofs)
end

sparse(S::SparseMatrixCOO) = sparse(S.I, S.J, S.V)
sparse(S::SparseMatrixCOO, m::Int, n::Int) = sparse(S.I, S.J, S.V, m, n)
sparse(S::SparseMatrixCOO, m::Int, n::Int, combine) = sparse(S.I, S.J, S.V, m, n, combine)
