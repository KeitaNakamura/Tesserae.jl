"""
    SparseMatrixCOO([N = 0])

Construct sparse matrix using COOrdinate format.

# Examples
```jldoctest
julia> S = SparseMatrixCOO();

julia> push!(S, [1.0 1.0; 1.0 1.0], [1, 2], [2, 3]);

julia> sparse!(S, 4, 4)
4×4 SparseMatrixCSC{Float64, Int64} with 4 stored entries:
  ⋅   1.0  1.0   ⋅
  ⋅   1.0  1.0   ⋅
  ⋅    ⋅    ⋅    ⋅
  ⋅    ⋅    ⋅    ⋅

julia> push!(S, [1.0, 2.0], [1, 2]); # Add diagonal entries

julia> sparse!(S)
2×3 SparseMatrixCSC{Float64, Int64} with 5 stored entries:
 1.0  1.0  1.0
  ⋅   3.0  1.0
```
"""
struct SparseMatrixCOO{T}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{T}
    csrrowptr::Vector{Int}
    csrcolval::Vector{Int}
    csrnzval::Vector{T}
    klasttouch::Vector{Int}
    csccolptr::Vector{Int}
    cscrowval::Vector{Int}
    cscnzval::Vector{T}
    dims::Vector{Int}
end

function SparseMatrixCOO{T}(N::Int = 0) where {T}
    I = Int[]
    J = Int[]
    V = T[]
    sizehint!(I, N)
    sizehint!(J, N)
    sizehint!(V, N)
    SparseMatrixCOO(I, J, V, Int[], Int[], T[], Int[], Int[], Int[], T[], [0,0])
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

# diagonal version
function Base.push!(S::SparseMatrixCOO, s, dofs::AbstractVector{Int})
    @assert length(s) == length(dofs)
    append!(S.V, s)
    append!(S.I, dofs)
    append!(S.J, dofs)
    S
end

function sparse!(S::SparseMatrixCOO, m::Int, n::Int, combine)
    coolen = length(S.I)
    resize!(S.klasttouch, n)
    resize!(S.csrrowptr, m+1)
    resize!(S.csrcolval, coolen)
    resize!(S.csrnzval, coolen)
    resize!(S.csccolptr, n+1)
    S.dims[1] = m
    S.dims[2] = n
    sparse!(S.I, S.J, S.V, m, n, combine, S.klasttouch, S.csrrowptr, S.csrcolval, S.csrnzval, S.csccolptr, S.cscrowval, S.cscnzval)
end
sparse!(S::SparseMatrixCOO, m::Int, n::Int) = sparse!(S, m, n, +)
sparse!(S::SparseMatrixCOO) = sparse!(S, SparseArrays.dimlub(S.I), SparseArrays.dimlub(S.J))

sparse(S::SparseMatrixCOO{T}, m::Int, n::Int) where {T} = SparseMatrixCSC{T,Int}(m, n, S.csccolptr, S.cscrowval, S.cscnzval)
sparse(S::SparseMatrixCOO) = sparse(S, S.dims[1], S.dims[2])

function Base.empty!(S::SparseMatrixCOO)
    empty!(S.I)
    empty!(S.J)
    empty!(S.V)
    S
end
