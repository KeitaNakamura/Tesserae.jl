struct MirroredHalo <: AxisRegion
    halo::Halo
end

@inline side(region::MirroredHalo) = side(region.halo)

@inline function mirror(region::Region, d::Int)
    replaceaxis(region, d, _mirror(axisregion(region, d)))
end

@inline _mirror(region::Halo) = MirroredHalo(region)
@inline _mirror(region::MirroredHalo) = region.halo
_mirror(::Boundary) = throw(ArgumentError("cannot mirror a Boundary axis because it is not necessarily a geometric fixed point"))
_mirror(::Physical) = throw(ArgumentError("cannot mirror a Physical axis because it does not identify a boundary side"))

@inline function _indexrange(region::MirroredHalo, physical::UnitRange{Int}, width::Int, node_aligned::Bool)
    s = side(region)
    s == -1 && return (first(physical) + width - 1 + node_aligned):-1:(first(physical) + node_aligned)
    s == +1 && return (last(physical) - node_aligned):-1:(last(physical) - width + 1 - node_aligned)
    throw(ArgumentError("side must be -1 or +1, got $s"))
end
