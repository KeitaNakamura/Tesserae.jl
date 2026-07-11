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
    offset::GridOffset{N}
end

function Region(placement::Placement, axes::Vararg{AxisRegion, N}; halo::Int) where {N}
    Region(placement, axes, halo, zero(GridOffset{N}))
end

@inline placement(region::Region) = region.placement
@inline axisregion(region::Region, d::Int) = region.axes[d]
@inline halo(region::Region) = region.halo
@inline nhalfsteps(region::Region, d::Int) = nhalfsteps(region.offset, d)
@inline side(region::Union{Ghost, Boundary}) = region.side

@inline function replaceaxis(region::Region, d::Int, axis::AxisRegion)
    Region(region.placement, Base.setindex(region.axes, axis, d), region.halo, region.offset)
end

@inline function shift(region::Region{N}, offset::GridOffset{N}) where {N}
    Region(region.placement, region.axes, region.halo, region.offset + offset)
end

@inline Base.:+(region::Region{N}, offset::GridOffset{N}) where {N} = shift(region, offset)
@inline Base.:+(offset::GridOffset{N}, region::Region{N}) where {N} = shift(region, offset)
@inline Base.:-(region::Region{N}, offset::GridOffset{N}) where {N} = shift(region, -offset)
Base.:-(::GridOffset, ::Region) = throw(ArgumentError("`GridOffset - Region` is not a translation; write `Region - GridOffset`"))

Base.broadcastable(region::Region) = Ref(region)
