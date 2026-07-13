"""
    AxisRegion

A one-dimensional region along an axis. An `N`-dimensional [`Region`](@ref) is
the Cartesian product of `N` axis regions. `full` includes the halo, while
`physical` excludes it.
"""
@enum AxisRegion::UInt8 begin
    full
    physical
    lowhalo
    highhalo
    lowboundary
    highboundary
end

@inline isfull(region::AxisRegion) = region == full
@inline isphysical(region::AxisRegion) = region == physical
@inline ishalo(region::AxisRegion) = region == lowhalo || region == highhalo
@inline isboundary(region::AxisRegion) = region == lowboundary || region == highboundary

@inline function side(region::AxisRegion)
    (region == lowhalo || region == lowboundary) && return -1
    (region == highhalo || region == highboundary) && return +1
    throw(ArgumentError("$region does not have a side"))
end

"""
    Region(location, axes...; halowidth)

An `N`-dimensional Cartesian product of axis regions. A region stores geometry
relative to the physical domain, but not concrete array indices or extents.
`halowidth` may be an `N`-tuple of axis widths or a scalar width shared by all
axes. Each width must be nonnegative. Shifting a region produces a
[`ShiftedRegion`](@ref).
"""
struct Region{N}
    location::Location
    axes::NTuple{N, AxisRegion}
    halowidth::NTuple{N, Int}
    function Region{N}(location::Location, axes::NTuple{N, AxisRegion}, halowidth::NTuple{N, Int}) where {N}
        N ≤ 8 * sizeof(UInt) || throw(ArgumentError("Region supports at most $(8 * sizeof(UInt)) dimensions"))
        valid_location = location == Cell() || location == Vertex() || any(d -> location == Face(d) || location == Edge(d), 1:N)
        valid_location || throw(ArgumentError("location is incompatible with a $N-dimensional Region"))
        all(width -> width ≥ 0, halowidth) || throw(ArgumentError("halowidth must be nonnegative, got $halowidth"))
        new{N}(location, axes, halowidth)
    end
end

function Region(location::Location, axes::NTuple{N, AxisRegion}, halowidth::NTuple{N, Int}) where {N}
    Region{N}(location, axes, halowidth)
end

function Region(location::Location, axes::NTuple{N, AxisRegion}; halowidth::Union{Int, NTuple{N, Int}}) where {N}
    widths = halowidth isa Int ? ntuple(_ -> halowidth, Val(N)) : halowidth
    Region(location, axes, widths)
end

function Region(location::Location, axes::Vararg{AxisRegion, N}; halowidth::Union{Int, NTuple{N, Int}}) where {N}
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
struct ShiftedRegion{N}
    region::Region{N}
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
