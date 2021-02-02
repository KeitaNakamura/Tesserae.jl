struct MPSpaceNearSurface
    nearsurface::BitVector
end

nearsurface(space::MPSpace) = MPSpaceNearSurface(space.nearsurface)
Base.any(space::MPSpaceNearSurface) = any(space.nearsurface)

struct NearSurfaceState{P}
    parent::P
    nearsurface::BitVector
end

Base.in(state, space::MPSpaceNearSurface) = NearSurfaceState(state, space.nearsurface)

function set!(dest::PointState, src::NearSurfaceState)
    set!(dest, src.parent, src.nearsurface)
    dest
end

function set!(dest::GridState, src::NearSurfaceState)
    set!(dest, src.parent, src.nearsurface)
    dest
end
