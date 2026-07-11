"""
    Placement

The phase of each logical grid axis, packed into a bit mask. A set bit denotes
a node-aligned axis; an unset bit denotes a cell-centered axis.
"""
struct Placement
    mask::UInt
end

@inline function axisbit(d::Int)
    1 ≤ d ≤ 8 * sizeof(UInt) || throw(ArgumentError("axis must be between 1 and $(8 * sizeof(UInt)), got $d"))
    one(UInt) << (d - 1)
end

"""
    Cell()

Create a placement that is cell-centered along every axis.
"""
Cell() = Placement(zero(UInt))

"""
    Face(d)

Create a placement that is node-aligned along axis `d` and cell-centered along
every other axis.
"""
Face(d::Int) = Placement(axisbit(d))

"""
    Edge(d)

Create a placement that is cell-centered along axis `d` and node-aligned along
every other axis. In three dimensions this is the center of an edge parallel to
axis `d`.
"""
Edge(d::Int) = Placement(~axisbit(d))

"""
    Vertex()

Create a placement that is node-aligned along every axis.
"""
Vertex() = Placement(typemax(UInt))

@inline isnodealigned(placement::Placement, d::Int) = !iszero(placement.mask & axisbit(d))
