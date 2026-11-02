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
Edge(d::Int) = Location(~Face(d).mask)
Vertex() = Location(typemax(UInt))

function canonical(location::Location, N::Int)
    1 ≤ N ≤ 8sizeof(UInt) || throw(ArgumentError("unsupported dimension $N"))
    active = N == 8sizeof(UInt) ? typemax(UInt) : (one(UInt) << N)-one(UInt)
    named = location == Vertex() || any(d -> location == Edge(d), 1:N)
    named || iszero(location.mask & ~active) ||
        throw(ArgumentError("placement is incompatible with $N dimensions"))
    Location(location.mask & active)
end
