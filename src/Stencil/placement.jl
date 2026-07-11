"""
    Placement

The phase of each logical grid axis, packed into a bit mask. A set bit denotes
a node-aligned axis; an unset bit denotes a cell-centered axis.
"""
struct Placement
    mask::UInt
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
Face(d::Int) = Placement(one(UInt) << (d - 1))

@inline isnodealigned(placement::Placement, d::Int) = !iszero(placement.mask & (one(UInt) << (d - 1)))
