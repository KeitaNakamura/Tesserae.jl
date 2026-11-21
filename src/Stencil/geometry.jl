"""
    AxisPlane(mesh, d, coordinate)

An infinite plane normal to axis `d`, at a physical coordinate in `mesh`'s
frame. Planes may be translated by grid offsets. Reflection is exact on the
sampling lattice; a plane that requires interpolation is rejected by `reflect`.
"""
struct AxisPlane{N, T, X, F<:GridFrame{N, T}}
    frame::F
    axis::Int
    coordinate::X  # Physical coordinate before translation.
    q0::Union{Int, Nothing}  # Quarter-cell index before translation, if on the lattice.
    shift::Int     # Translation in half-cell units.
end

AxisPlane(mesh::CartesianMesh, d::Int, x::Real) = AxisPlane(GridFrame(mesh), d, x)
function AxisPlane(frame::GridFrame{N}, d::Int, x::Real) where {N}
    1 ≤ d ≤ N || throw(ArgumentError("invalid plane axis $d"))
    isfinite(x) || throw(ArgumentError("the plane coordinate must be finite"))
    AxisPlane(frame, d, x, latticequarter(frame, d, x), 0)
end

# Reflection maps samples back to the staggered lattice at quarter-cell planes.
function quarterpoint(frame::GridFrame{N, T}, d, q) where {N, T}
    if iseven(q)
        i = fld(q, 4)+1
        return only(coordinate_axis(i:i, iszero(mod(q, 4)), frame, d))
    end
    origin, h = gridbasis(frame.origin[d], frame.spacing)
    T(origin+(q/4)*h)
end

function nearestquarter(frame::GridFrame, d, x)
    origin, h = frame.origin[d], frame.spacing
    W = promote_type(Float64, typeof(x), typeof(h))
    q = 4*((W(x)-W(origin))/W(h))
    typemin(Int) < q < typemax(Int) ? round(Int, q, RoundNearestTiesUp) : nothing
end

function latticequarter(frame::GridFrame, d, x)
    q = nearestquarter(frame, d, x)
    isnothing(q) && return nothing
    point = quarterpoint(frame, d, q)
    x == point && return q
    if x isa AbstractFloat && x == oftype(x, point)
        # A lower-precision input must identify just one sample.
        lower = oftype(x, quarterpoint(frame, d, q-1))
        upper = oftype(x, quarterpoint(frame, d, q+1))
        lower < x < upper && return q
    end
    nothing
end

quarterposition(p::AxisPlane, q=p.q0) = isnothing(q) ? nothing : q+2*p.shift

# Off-lattice planes in different quarter-cell intervals get different hash keys.
function positionkey(p::AxisPlane)
    isnothing(p.q0) || return (quarterposition(p), true)
    near = nearestquarter(p.frame, p.axis, p.coordinate)
    (quarterposition(p, near), false)
end

# Check the displacement in both directions; rounding a large sum can lose
# the smaller coordinate even when the difference looks like a whole cell.
function separatedby(x, y, halfcells, h)
    iszero(halfcells) && return x == y
    delta = (halfcells/2)*h
    x != y && x-delta == y && y+delta == x
end

function sameposition(a::AxisPlane, b::AxisPlane)
    pos = positionkey(a)
    pos == positionkey(b) || return false
    _, aligned = pos
    aligned || separatedby(a.coordinate, b.coordinate, b.shift-a.shift, a.frame.spacing)
end
Base.:(==)(a::AxisPlane, b::AxisPlane) = a.frame == b.frame && a.axis == b.axis && sameposition(a, b)
Base.isequal(a::AxisPlane, b::AxisPlane) = isequal(a.frame, b.frame) && a.axis == b.axis && sameposition(a, b)
Base.hash(p::AxisPlane, h::UInt) = hash((p.frame, p.axis, positionkey(p)), h)

"""Return the lower and upper physical planes of `mesh` along axis `d`."""
function boundaries(mesh::CartesianMesh{N}, d::Int) where {N}
    1 ≤ d ≤ N || throw(ArgumentError("invalid plane axis $d"))
    frame = GridFrame(mesh)
    left = AxisPlane(frame, d, frame.origin[d], 0, 0)
    (left, left+(size(mesh, d)-1)*unitoffsets(Val(N))[d])
end

Base.:+(p::AxisPlane{N}, e::GridOffset{N}) where {N} =
    AxisPlane(p.frame, p.axis, p.coordinate, p.q0, p.shift+e.doubled[p.axis])
Base.:+(e::GridOffset, p::AxisPlane) = p+e
Base.:-(p::AxisPlane{N}, e::GridOffset{N}) where {N} = p+(-e)

function Base.:-(a::AxisPlane{N}, b::AxisPlane{N}) where {N}
    checkframe(a.frame, b.frame)
    a.axis == b.axis || throw(ArgumentError("plane displacement requires parallel planes"))
    aq, bq = quarterposition(a), quarterposition(b)
    if !isnothing(aq) && !isnothing(bq)
        distance = lattice_integer((aq-bq)//2)
    else
        x, y, h = a.coordinate, b.coordinate, a.frame.spacing
        n = lattice_integer(2*((x-y)/h))
        separatedby(x, y, n, h) || throw(ArgumentError("planes do not differ by a half-cell displacement"))
        distance = n+a.shift-b.shift
    end
    GridOffset(ntuple(d -> d == a.axis ? distance : 0, Val(N)))
end

"""
    reflect(r, plane)

Reflect every selected position through `plane`, retaining point correspondence.
The reflected axis reverses order. Applying the same reflection twice restores
both positions and order. No interpolation or field values are involved.
"""
function reflect(r::Region{N}, p::AxisPlane{N}) where {N}
    d = p.axis
    phase = !isnodealigned(r.location, d)
    checkframe(r.frame, p.frame)
    q = quarterposition(p)
    isnothing(q) && throw(ArgumentError("reflection requires a plane on the quarter-cell lattice"))
    mirror = q + (phase ? 2 : 4)
    mask = r.location.mask ⊻ (UInt(isodd(mirror)) << (d-1))
    indices = ntuple(Val(N)) do j
        j == d || return r.indices[j]
        fld(mirror+phase, 2) .- r.indices[d]
    end
    Region(Location(mask), r.cells, indices, r.frame)
end

# For an off-lattice plane, compare the stored coordinate with the samples
# shifted back by the plane's translation. This preserves either side of a node.
function planeindices(r::Region, p::AxisPlane)
    checkframe(r.frame, p.frame)
    d = p.axis
    indices = r.indices[d]
    descending = step(indices) < 0
    phase = isnodealigned(r.location, d) ? 0 : 2
    q = quarterposition(p)
    if !isnothing(q)
        return searchsorted(indices, (q-phase)//4+1; rev=descending)
    end
    point(i::Int) = quarterpoint(r.frame, d, 4*(i-1)+phase-2*p.shift)
    point(::AxisPlane) = p.coordinate
    searchsorted(indices, p; by=point, rev=descending)
end

function Base.intersect(r::Region{N}, p::AxisPlane{N}) where {N}
    d = p.axis
    axis = r.indices[d]
    on = planeindices(r, p)
    indices = Base.setindex(r.indices, axis[on], d)
    Region(r.location, r.cells, indices, r.frame)
end
Base.intersect(p::AxisPlane, r::Region) = intersect(r, p)

"""
    split(r, plane)

Return the negative-side, on-plane and positive-side sample selections. They
are disjoint and cover `r`; each keeps its original order and placement.
"""
function Base.split(r::Region{N}, p::AxisPlane{N}) where {N}
    d, axis = p.axis, r.indices[p.axis]
    descending = step(axis) < 0
    on = planeindices(r, p)
    parts = map((1:first(on)-1, on, last(on)+1:length(axis))) do positions
        indices = Base.setindex(r.indices, axis[positions], d)
        Region(r.location, r.cells, indices, r.frame)
    end
    descending ? reverse(parts) : parts
end

"""
    halos(grid, plane)
    halos(array, physical_region, plane)

Select all stored samples outside one physical boundary. The physical samples
are located using the same rules as `A[physical_region]`. An array with only
physical samples has an empty halo even when its axes start elsewhere.
Transverse axes include stored corners. The plane must bound the physical
domain. The caller writes the boundary equations and chooses their order.
"""
function halos(A::AbstractArray{T, N}, r::Region{N}, p::AxisPlane{N}) where {T, N}
    iswhole(r) || throw(ArgumentError("halo selection requires the whole physical region"))
    checkframe(r.frame, p.frame)
    storage = axes(A)
    physical = regionranges(r, storage)
    d = p.axis
    side = quarterposition(p)
    side in (0, 4r.cells[d]) || throw(ArgumentError("the plane must be a physical boundary of this region"))
    indices = ntuple(Val(N)) do j
        lo = 1-(first(physical[j])-first(storage[j]))
        hi = size(r, j)+(last(storage[j])-last(physical[j]))
        j != d && return lo:hi
        side == 0 ? (lo:0) : ((size(r, d)+1):hi)
    end
    Region(r.location, r.cells, indices, r.frame)
end
halos(grid::Grid, p::AxisPlane) = halos(grid, Region(grid), p)
