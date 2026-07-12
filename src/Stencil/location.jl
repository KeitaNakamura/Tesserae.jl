"""
    Location

The phase of each logical grid axis, packed into a bit mask. A set bit denotes
a node-aligned axis; an unset bit denotes a cell-centered axis.
"""
struct Location
    mask::UInt
end

@inline function axisbit(d::Int)
    1 ≤ d ≤ 8 * sizeof(UInt) || throw(ArgumentError("axis must be between 1 and $(8 * sizeof(UInt)), got $d"))
    one(UInt) << (d - 1)
end

"""
    Cell()

Create a location that is cell-centered along every axis.
"""
Cell() = Location(zero(UInt))

"""
    Face(d)

Create a location that is node-aligned along axis `d` and cell-centered along
every other axis.
"""
Face(d::Int) = Location(axisbit(d))

"""
    Edge(d)

Create a location that is cell-centered along axis `d` and node-aligned along
every other axis. In three dimensions this is the center of an edge parallel to
axis `d`.
"""
Edge(d::Int) = Location(~axisbit(d))

"""
    Vertex()

Create a location that is node-aligned along every axis.
"""
Vertex() = Location(typemax(UInt))

@inline isnodealigned(location::Location, d::Int) = !iszero(location.mask & axisbit(d))
