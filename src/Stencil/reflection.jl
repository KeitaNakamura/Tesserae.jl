struct ReflectionMap{N, R <: Region{N}} <: RegionMap
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
Reflection is defined only for an unshifted [`Region`](@ref). Reflecting the
same axis twice restores its original index order.
"""
@inline function reflect(region::Region{N}, d::Int) where {N}
    1 ≤ d ≤ N || throw(ArgumentError("axis must be between 1 and $N, got $d"))
    map(axis -> axis isa Halo, axisregions(region))[d] || throw(ArgumentError("only a Halo axis can be reflected"))
    ReflectionMap(region, axisbit(d))
end

reflect(::ShiftedRegion, ::Int) = throw(ArgumentError("a ShiftedRegion cannot be reflected"))

@inline function reflect(reflection::ReflectionMap{N}, d::Int) where {N}
    region = reflection.region
    1 ≤ d ≤ N || throw(ArgumentError("axis must be between 1 and $N, got $d"))
    map(axis -> axis isa Halo, axisregions(region))[d] || throw(ArgumentError("only a Halo axis can be reflected"))
    mask = xor(reflection.mask, axisbit(d))
    ReflectionMap(region, mask)
end

function mappedranges(reflection::ReflectionMap{N}, array_axes::NTuple{N, AbstractUnitRange{Int}}) where {N}
    region = reflection.region
    ranges = indexranges(region, array_axes)
    dimensions = ntuple(identity, Val(N))

    map(axisregions(region), ranges, array_axes, halowidth(region), dimensions) do axis, range, array_axis, width, d
        if isreflected(reflection, d)
            ncells = length(array_axis) - 2 * width - isnodealigned(placement(region), d)
            width ≤ ncells || throw(DimensionMismatch("halowidth $width exceeds $ncells physical cells on axis $d"))
        end
        _mappedrange(reflection, d, axis, range, isnodealigned(placement(region), d))
    end
end

@inline _mappedrange(::ReflectionMap, ::Int, ::Union{Physical, Boundary}, range::UnitRange{Int}, ::Bool) = range

@inline function _mappedrange(reflection::ReflectionMap, d::Int, axis::Halo, range::UnitRange{Int}, node_aligned::Bool)
    isreflected(reflection, d) && return _reflectionrange(axis, range, node_aligned)
    first(range):1:last(range)
end

@inline function _reflectionrange(region::Halo, halo::UnitRange{Int}, node_aligned::Bool)
    if side(region) == -1
        (last(halo) + length(halo) + node_aligned):-1:(last(halo) + 1 + node_aligned)
    else
        (first(halo) - 1 - node_aligned):-1:(first(halo) - length(halo) - node_aligned)
    end
end
