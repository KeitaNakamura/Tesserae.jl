"""Sampling placement on a Cartesian grid; set bits denote face-aligned axes."""
struct Location
    mask::UInt
end

"""`Cell()` places a quantity at cell centers."""
Cell() = Location(zero(UInt))

"""`Face(d)` places a quantity at the centers of faces normal to axis `d`."""
function Face(d::Int)
    1 ≤ d ≤ 8sizeof(UInt) || throw(ArgumentError("invalid face axis $d"))
    Location(one(UInt) << (d - 1))
end

@inline isnodealigned(location::Location, d::Int) = !iszero(location.mask & (one(UInt) << (d - 1)))

# Named placements use the same per-axis grid-line/center representation.
"""`Edge(d)` places a quantity on edges parallel to axis `d` in three dimensions."""
Edge(d::Int) = Location(~Face(d).mask)
"""`Vertex()` places a quantity at grid vertices in every dimension."""
Vertex() = Location(typemax(UInt))

function canonical(location::Location, N::Int)
    1 ≤ N ≤ 8sizeof(UInt) || throw(ArgumentError("unsupported dimension $N"))
    active = typemax(UInt) >> (8sizeof(UInt)-N)
    named = location == Vertex() || any(d -> location == Edge(d), 1:N)
    named || iszero(location.mask & ~active) ||
        throw(ArgumentError("placement is incompatible with $N dimensions"))
    Location(location.mask & active)
end

# Twice the displacement is stored as integers, making all half-cell shifts exact.
struct GridOffset{N}
    doubled::NTuple{N, Int}
end

"""
    unitoffsets(Val(N))
    unitoffsets(r::Region)

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

"""
    location + offset
    location - offset

Sampling placement after a grid shift, in the dimension of `offset`. Half-cell
shifts exchange center and grid-line alignment along the shifted axes. A whole
cell shift preserves placement. This changes a placement pattern, not a finite
Region's selected points; the result can be passed to `generate_grid`.
"""
function Base.:+(location::Location, offset::GridOffset{N}) where {N}
    mask = canonical(location, N).mask
    for d in 1:N
        isodd(offset.doubled[d]) && (mask ⊻= one(UInt) << (d-1))
    end
    Location(mask)
end
Base.:+(offset::GridOffset, location::Location) = location+offset
Base.:-(location::Location, offset::GridOffset) = location+offset
