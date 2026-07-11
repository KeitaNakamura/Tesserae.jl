struct MirroredGhost <: AxisRegion
    ghost::Ghost
end

@inline side(region::MirroredGhost) = side(region.ghost)

@inline function mirror(region::Region, d::Int)
    replaceaxis(region, d, _mirror(axisregion(region, d)))
end

@inline _mirror(region::Ghost) = MirroredGhost(region)
@inline _mirror(region::MirroredGhost) = region.ghost
_mirror(::Boundary) = throw(ArgumentError("cannot mirror a Boundary axis because it is not necessarily a geometric fixed point"))
_mirror(::Physical) = throw(ArgumentError("cannot mirror a Physical axis because it does not identify a boundary side"))

@inline function _indexrange(region::MirroredGhost, physical::UnitRange{Int}, halo::Int, node_aligned::Bool)
    s = side(region)
    s == -1 && return (first(physical) + halo - 1 + node_aligned):-1:(first(physical) + node_aligned)
    s == +1 && return (last(physical) - node_aligned):-1:(last(physical) - halo + 1 - node_aligned)
    throw(ArgumentError("side must be -1 or +1, got $s"))
end
