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

function SparseMatrixCOO{T}(N::Int = 0) where {T}
    I = Int[]
    J = Int[]
    V = T[]
    sizehint!(I, N)
    sizehint!(J, N)
    sizehint!(V, N)
    SparseMatrixCOO(I, J, V)
end
SparseMatrixCOO(N::Int = 0) = SparseMatrixCOO{Float64}(N)

function Base.push!(S::SparseMatrixCOO, s, I::AbstractVector{Int}, J::AbstractVector{Int})
    @assert size(s) == (length(I), length(J))
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

function Base.push!(S::SparseMatrixCOO, s, dofs::AbstractVector{Int})
    push!(S, s, dofs, dofs)
end

sparse(S::SparseMatrixCOO) = sparse(S.I, S.J, S.V)
sparse(S::SparseMatrixCOO, m::Int, n::Int) = sparse(S.I, S.J, S.V, m, n)
sparse(S::SparseMatrixCOO, m::Int, n::Int, combine) = sparse(S.I, S.J, S.V, m, n, combine)

function Base.empty!(S::SparseMatrixCOO)
    empty!(S.I)
    empty!(S.J)
    empty!(S.V)
    S
end
