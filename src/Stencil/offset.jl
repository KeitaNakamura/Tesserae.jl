# Twice the displacement is stored as integers, making all half-cell shifts exact.
struct GridOffset{N}
    doubled::NTuple{N,Int}
end

"""
    unitoffsets(Val(N))

Return the `N` coordinate directions, each representing one cell width.
Adding `e/2` to a region moves its sampling placement by half a cell.
"""
function unitoffsets(::Val{N}) where {N}
    ntuple(Val(N)) do d
        GridOffset(ntuple(j -> 2Int(j == d), Val(N)))
    end
end

@inline Base.:+(a::GridOffset{N}, b::GridOffset{N}) where {N} = GridOffset(map(+, a.doubled, b.doubled))
@inline Base.:-(a::GridOffset{N}, b::GridOffset{N}) where {N} = GridOffset(map(-, a.doubled, b.doubled))
@inline Base.:-(a::GridOffset) = GridOffset(map(-, a.doubled))
@inline Base.:*(n::Int, a::GridOffset) = GridOffset(map(x -> n*x, a.doubled))
@inline Base.:*(a::GridOffset, n::Int) = n*a

function Base.:/(a::GridOffset, n::Int)
    iszero(n) && throw(DivideError())
    all(x -> iszero(rem(x, n)), a.doubled) ||
        throw(ArgumentError("displacement must be a multiple of half a cell"))
    GridOffset(map(x -> div(x, n), a.doubled))
end

Base.zero(::Type{GridOffset{N}}) where {N} = GridOffset(ntuple(_ -> 0, Val(N)))
Base.broadcastable(a::GridOffset) = Ref(a)
