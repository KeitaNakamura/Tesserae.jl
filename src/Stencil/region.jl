"""
    AxisRegion

A one-dimensional region along an axis. An `N`-dimensional [`Region`](@ref) is
the Cartesian product of `N` axis regions.
"""
abstract type AxisRegion end

"""
    Physical()

The physical part of an axis.
"""
struct Physical <: AxisRegion end

"""
    Ghost(side)

The ghost part of the low (`side = -1`) or high (`side = +1`) side of an axis.
"""
struct Ghost <: AxisRegion
    side::Int
end

"""
    Boundary(side)

The physical boundary on the low (`side = -1`) or high (`side = +1`) side of
an axis.
"""
struct Boundary <: AxisRegion
    side::Int
end

"""
    Region(placement, axes...; halo)

An `N`-dimensional Cartesian product of axis regions. A region stores geometry
relative to the physical domain, but not concrete array indices or extents.
"""
struct Region{N, Axes <: NTuple{N, AxisRegion}}
    placement::Placement
    axes::Axes
    halo::Int
end

function Region(placement::Placement, axes::Vararg{AxisRegion, N}; halo::Int) where {N}
    Region{N, typeof(axes)}(placement, axes, halo)
end
