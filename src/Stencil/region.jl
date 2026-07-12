"""
    AxisRegion

A one-dimensional region along an axis. An `N`-dimensional [`Region`](@ref) is
the Cartesian product of `N` axis regions.
"""
abstract type AxisRegion end

"""
    Full()

The full extent of an array axis, including its halo.
"""
struct Full <: AxisRegion end

"""
    Physical()

The full non-halo extent of an axis, including its boundaries.
"""
struct Physical <: AxisRegion end

"""
    Halo(side)

The halo region on the low (`side = -1`) or high (`side = +1`) side of an axis.
"""
struct Halo <: AxisRegion
    side::Int
    function Halo(side::Int)
        side == -1 || side == +1 || throw(ArgumentError("side must be -1 or +1, got $side"))
        new(side)
    end
end

"""
    Boundary(side)

The physical boundary on the low (`side = -1`) or high (`side = +1`) side of
an axis.
"""
struct Boundary <: AxisRegion
    side::Int
    function Boundary(side::Int)
        side == -1 || side == +1 || throw(ArgumentError("side must be -1 or +1, got $side"))
        new(side)
    end
end

@inline side(region::Union{Halo, Boundary}) = region.side

"""
    Region(location, axes...; halowidth)

An `N`-dimensional Cartesian product of axis regions. A region stores geometry
relative to the physical domain, but not concrete array indices or extents.
`halowidth` may be an `N`-tuple of axis widths or a scalar width shared by all
axes. Each width must be nonnegative. Shifting a region produces a
[`ShiftedRegion`](@ref).
"""
struct Region{N, Axes <: NTuple{N, AxisRegion}}
    location::Location
    axes::Axes
    halowidth::NTuple{N, Int}
    function Region{N, Axes}(location::Location, axes::Axes, halowidth::NTuple{N, Int}) where {N, Axes <: NTuple{N, AxisRegion}}
        N ≤ 8 * sizeof(UInt) || throw(ArgumentError("Region supports at most $(8 * sizeof(UInt)) dimensions"))
        valid_location = location == Cell() || location == Vertex() || any(d -> location == Face(d) || location == Edge(d), 1:N)
        valid_location || throw(ArgumentError("location is incompatible with a $N-dimensional Region"))
        all(width -> width ≥ 0, halowidth) || throw(ArgumentError("halowidth must be nonnegative, got $halowidth"))
        new{N, Axes}(location, axes, halowidth)
    end
end

function Region(location::Location, axes::Axes, halowidth::NTuple{N, Int}) where {N, Axes <: NTuple{N, AxisRegion}}
    Region{N, Axes}(location, axes, halowidth)
end

function Region(location::Location, axes::NTuple{N, AxisRegion}; halowidth::Union{Int, NTuple{N, Int}}) where {N}
    widths = halowidth isa Int ? ntuple(_ -> halowidth, Val(N)) : halowidth
    Region(location, axes, widths)
end

function Region(location::Location, axes::NTuple{N, Union{AxisRegion, Colon}}; halowidth::Union{Int, NTuple{N, Int}}) where {N}
    normalized = map(axis -> axis isa Colon ? Full() : axis, axes)
    Region(location, normalized; halowidth)
end

function Region(location::Location, axes::Vararg{Union{AxisRegion, Colon}, N}; halowidth::Union{Int, NTuple{N, Int}}) where {N}
    Region(location, axes; halowidth)
end

@inline location(region::Region) = region.location
@inline axisregions(region::Region) = region.axes
@inline halowidth(region::Region) = region.halowidth
@inline halowidth(region::Region, d::Int) = region.halowidth[d]

@inline function shift(region::Region{N}, displacement::GridOffset{N}) where {N}
    ShiftedRegion(region, displacement)
end

@inline Base.:+(region::Region{N}, offset::GridOffset{N}) where {N} = shift(region, offset)
@inline Base.:+(offset::GridOffset{N}, region::Region{N}) where {N} = shift(region, offset)
@inline Base.:-(region::Region{N}, offset::GridOffset{N}) where {N} = shift(region, -offset)
Base.:-(::GridOffset, ::Region) = throw(ArgumentError("`GridOffset - region` is not a shift; write `region - GridOffset`"))

Base.broadcastable(region::Region) = Ref(region)

"""
    ShiftedRegion

A [`Region`](@ref) shifted by a [`GridOffset`](@ref). Shifted regions are
created by adding or subtracting a grid offset from a region.
"""
struct ShiftedRegion{N, R <: Region{N}}
    region::R
    offset::GridOffset{N}
end

@inline Base.parent(shifted::ShiftedRegion) = shifted.region
@inline offset(shifted::ShiftedRegion) = shifted.offset

@inline function shift(shifted::ShiftedRegion{N}, displacement::GridOffset{N}) where {N}
    ShiftedRegion(parent(shifted), offset(shifted) + displacement)
end

@inline Base.:+(shifted::ShiftedRegion{N}, offset::GridOffset{N}) where {N} = shift(shifted, offset)
@inline Base.:+(offset::GridOffset{N}, shifted::ShiftedRegion{N}) where {N} = shift(shifted, offset)
@inline Base.:-(shifted::ShiftedRegion{N}, offset::GridOffset{N}) where {N} = shift(shifted, -offset)
Base.:-(::GridOffset, ::ShiftedRegion) = throw(ArgumentError("`GridOffset - region` is not a shift; write `region - GridOffset`"))

Base.broadcastable(shifted::ShiftedRegion) = Ref(shifted)

abstract type RegionMap end

Base.broadcastable(map::RegionMap) = Ref(map)
