struct ReflectionMap{R <: Region} <: RegionMap
    region::R
    mask::UInt
end

@inline isreflected(reflection::ReflectionMap, d::Int) = !iszero(reflection.mask & axisbit(d))

"""
    reflect(region, d)
    reflect(reflection, d)

Reflect halo axis `d` across its physical boundary. Reflection is defined by
the placement of the axis:

    Cell: …  b  a  |  a  b  …
    Face: …  c  b (a) b  c  …

Here `|` lies between cell-centered samples, while `(a)` lies on the boundary.
Reflecting the same axis twice restores its original index order.
"""
@inline function reflect(region::Region, d::Int)
    axisregion(region, d) isa Halo || throw(ArgumentError("only a Halo axis can be reflected"))
    ReflectionMap(region, axisbit(d))
end

@inline function reflect(reflection::ReflectionMap, d::Int)
    axisregion(reflection.region, d) isa Halo || throw(ArgumentError("only a Halo axis can be reflected"))
    mask = xor(reflection.mask, axisbit(d))
    ReflectionMap(reflection.region, mask)
end

function mappedranges(reflection::ReflectionMap{R}, array_axes::NTuple{N, AbstractUnitRange{Int}}) where {N, R <: Region{N}}
    region = reflection.region
    ranges = indexranges(region, array_axes)
    _mappedranges(reflection, placement(region), axisregions(region), ranges, 1)
end

@inline _mappedrange(::ReflectionMap, ::Int, ::Union{Physical, Boundary}, range::UnitRange{Int}, ::Bool) = range

@inline function _mappedrange(reflection::ReflectionMap, d::Int, axis::Halo, range::UnitRange{Int}, node_aligned::Bool)
    isreflected(reflection, d) && return _reflectionrange(axis, range, node_aligned)
    first(range):1:last(range)
end

@inline _mappedranges(::ReflectionMap, ::Placement, ::Tuple{}, ::Tuple{}, ::Int) = ()

@inline function _mappedranges(reflection::ReflectionMap, placement::Placement, axes::Tuple{A, Vararg}, ranges::Tuple{I, Vararg}, d::Int) where {A, I}
    axis = first(axes)
    range = first(ranges)
    mapped = _mappedrange(reflection, d, axis, range, isnodealigned(placement, d))
    (mapped, _mappedranges(reflection, placement, Base.tail(axes), Base.tail(ranges), d + 1)...)
end

@inline function _reflectionrange(region::Halo, halo::UnitRange{Int}, node_aligned::Bool)
    s = side(region)
    s == -1 && return (last(halo) + length(halo) + node_aligned):-1:(last(halo) + 1 + node_aligned)
    s == +1 && return (first(halo) - 1 - node_aligned):-1:(first(halo) - length(halo) - node_aligned)
    throw(ArgumentError("side must be -1 or +1, got $s"))
end
