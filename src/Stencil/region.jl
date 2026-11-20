const IndexRange = StepRange{Int, Int}

struct Region{N, T, F<:GridFrame{N, T}}
    location::Location
    cells::NTuple{N, Int}
    indices::NTuple{N, IndexRange}
    frame::F
    function Region(location::Location, cells::NTuple{N, Int}, indices::NTuple{N, AbstractRange{<:Integer}}, frame::F) where {N, T, F<:GridFrame{N, T}}
        location = canonical(location, N)
        all(>(0), cells) || throw(ArgumentError("the region must span a positive number of cells"))
        all(r -> abs(step(r)) == 1, indices) || throw(ArgumentError("region steps must be +1 or -1"))
        new{N, T, F}(location, cells, map(IndexRange, indices), frame)
    end
end

function samplecounts(location::Location, cells::NTuple{N, Int}) where {N}
    ntuple(d -> cells[d]+isnodealigned(location, d), Val(N))
end

function whole_region(location::Location, cells::NTuple{N, Int}, frame::GridFrame{N}) where {N}
    indices = map(Base.OneTo, samplecounts(location, cells))
    Region(location, cells, indices, frame)
end

"""Create a logical whole-domain region from the sample counts at this placement."""
function Region(location::Location, samples::NTuple{N, Int}) where {N}
    cells = ntuple(d -> samples[d]-Int(isnodealigned(location, d)), Val(N))
    whole_region(location, cells, GridFrame(Val(N)))
end

"""Create a physical sampling region in the coordinate frame of `mesh`."""
Region(location::Location, mesh::CartesianMesh) = whole_region(location, map(n -> n-1, size(mesh)), GridFrame(mesh))

function iswhole(r::Region)
    r.indices == map(Base.OneTo, samplecounts(r.location, r.cells))
end

"""Change placement on the same whole domain. This does not interpolate data."""
function Region(location::Location, source::Region)
    iswhole(source) || throw(ArgumentError("placement conversion requires a whole-domain region; use shifts for adjacent samples"))
    whole_region(location, source.cells, source.frame)
end

function Base.:(==)(a::Region, b::Region)
    a.location == b.location && a.cells == b.cells && a.indices == b.indices && a.frame == b.frame
end
function Base.isequal(a::Region, b::Region)
    isequal(a.location, b.location) && isequal(a.cells, b.cells) &&
        isequal(a.indices, b.indices) && isequal(a.frame, b.frame)
end
Base.hash(r::Region, h::UInt) = hash((r.location, r.cells, r.indices, r.frame), h)

function Base.size(r::Region{N}, d::Integer) where {N}
    d ≥ 1 || throw(BoundsError(size(r), d))
    d > N ? 1 : length(r.indices[d])
end
Base.size(r::Region) = map(length, r.indices)
Base.ndims(::Region{N}) where {N} = N
unitoffsets(::Region{N}) where {N} = unitoffsets(Val(N))

function select_interior(r::Region{N}, dimensions::Tuple{Vararg{Int}}) where {N}
    all(d -> 1 ≤ d ≤ N, dimensions) || throw(ArgumentError("invalid interior axis"))
    indices = ntuple(Val(N)) do d
        d in dimensions || return r.indices[d]
        physical = isnodealigned(r.location, d) ? (2:r.cells[d]) : (1:r.cells[d])
        intersect(r.indices[d], physical)
    end
    Region(r.location, r.cells, indices, r.frame)
end

"""
    interior(r)
    interior(r, d)

Select sample points strictly inside the physical domain, preserving their order
and placement. With `d`, restrict only that axis. Cell centers remain included;
samples on physical walls and outside the physical domain are excluded.
"""
interior(r::Region{N}) where {N} = select_interior(r, ntuple(identity, Val(N)))
interior(r::Region, d::Int) = select_interior(r, (d,))

"""
    boundaries(r, d)
    boundaries(r)

Select samples on the lower and upper physical walls, retaining singleton
dimensions, placement, transverse selections and order. The placement must have
samples on these walls. A wall absent from the selection gives an empty region.
"""
function boundaries(r::Region{N}, d::Int) where {N}
    1 ≤ d ≤ N || throw(ArgumentError("invalid boundary axis $d"))
    isnodealigned(r.location, d) || throw(ArgumentError("this placement has no samples on these physical walls"))
    map((1, r.cells[d]+1)) do side
        indices = ntuple(j -> j == d ? intersect(r.indices[j], side:side) : r.indices[j], Val(N))
        Region(r.location, r.cells, indices, r.frame)
    end
end

function boundaries(r::Region)
    count_ones(r.location.mask) == 1 || throw(ArgumentError("specify the boundary axis when it is not unique"))
    boundaries(r, trailing_zeros(r.location.mask)+1)
end

function shift_region(r::Region{N}, e::GridOffset{N}, sign::Int) where {N}
    indices = ntuple(Val(N)) do d
        phase = !isnodealigned(r.location, d) + sign*e.doubled[d]
        r.indices[d] .+ fld(phase, 2)
    end
    Region(r.location+e, r.cells, indices, r.frame)
end
Base.:+(r::Region, e::GridOffset) = shift_region(r, e, 1)
Base.:+(e::GridOffset{N}, r::Region{N}) where {N} = r+e
Base.:-(r::Region, e::GridOffset) = shift_region(r, e, -1)
Base.broadcastable(r::Region) = Ref(r)

"""Intersect sample positions, retaining the first region's order and lattice."""
function Base.intersect(a::Region{N}, b::Region{N}) where {N}
    checkframe(a.frame, b.frame)
    indices = ntuple(Val(N)) do d
        if isnodealigned(a.location, d) != isnodealigned(b.location, d)
            return a.indices[d][1:0]
        end
        intersect(a.indices[d], b.indices[d])
    end
    Region(a.location, a.cells, indices, a.frame)
end
