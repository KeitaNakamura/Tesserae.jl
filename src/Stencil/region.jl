struct Region{N}
    location::Location
    cells::NTuple{N,Int}
    indices::NTuple{N,UnitRange{Int}}
    function Region{N}(location::Location, cells::NTuple{N,Int},
                       indices::NTuple{N,UnitRange{Int}}) where {N}
        location = canonical(location, N)
        all(>(0), cells) ||
            throw(ArgumentError("the region must span a positive number of cells"))
        all(d -> isempty(indices[d]) ||
            1 ≤ first(indices[d]) ≤ last(indices[d]) ≤ cells[d]+Int(isnodealigned(location, d)), 1:N) ||
            throw(ArgumentError("selected indices must lie within the whole-domain placement"))
        normalized = map(i -> isempty(i) ? (1:0) : i, indices)
        new{N}(location, cells, normalized)
    end
end

"""Create a whole-domain region from the number of samples at this placement."""
function Region(location::Location, samples::NTuple{N,Int}) where {N}
    location = canonical(location, N)
    cells = ntuple(d -> samples[d]-Int(isnodealigned(location, d)), Val(N))
    Region{N}(location, cells, ntuple(d -> 1:samples[d], Val(N)))
end

function iswhole(r::Region{N}) where {N}
    all(d -> r.indices[d] == (1:(r.cells[d]+Int(isnodealigned(r.location, d)))), 1:N)
end

"""Change placement on the same whole domain. This does not interpolate data."""
function Region(location::Location, source::Region{N}) where {N}
    iswhole(source) || throw(ArgumentError(
        "placement conversion currently requires a whole-domain region; use shifts for adjacent samples"))
    Region(location, ntuple(d -> source.cells[d]+Int(isnodealigned(location, d)), Val(N)))
end

sampleindices(r::Region, d::Int) = r.indices[d]

function Base.size(r::Region{N}, d::Integer) where {N}
    d ≥ 1 || throw(BoundsError(size(r), d))
    d > N && return 1
    length(sampleindices(r, Int(d)))
end
Base.size(r::Region{N}) where {N} = ntuple(d -> size(r, d), Val(N))
Base.ndims(::Region{N}) where {N} = N
unitoffsets(::Region{N}) where {N} = unitoffsets(Val(N))

intersect_range(a::UnitRange{Int}, b::UnitRange{Int}) = max(first(a), first(b)):min(last(a), last(b))

function select_interior(r::Region{N}, dimensions::Tuple{Vararg{Int}}) where {N}
    all(d -> 1 ≤ d ≤ N, dimensions) || throw(ArgumentError("invalid interior axis"))
    indices = ntuple(Val(N)) do d
        d in dimensions && isnodealigned(r.location, d) ? intersect_range(r.indices[d], 2:r.cells[d]) : r.indices[d]
    end
    Region{N}(r.location, r.cells, indices)
end

"""
    interior(r)
    interior(r, d)

Exclude evaluation points on physical boundaries, preserving the placement.
Without `d`, exclude boundaries in every direction. With `d`, exclude only
boundaries normal to that axis, preserving existing selections in other axes.
Cell centers remain included because they are not on the physical boundary.
Selections are intersected with the existing region: applying `interior` in
the direction normal to a selected physical boundary produces an empty region.
"""
interior(r::Region{N}) where {N} = select_interior(r, ntuple(identity, Val(N)))
interior(r::Region, d::Int) = select_interior(r, (d,))

"""
    validregion(r, offsets::Tuple)
    validregion(r, offset, more_offsets...)

Select the points of `r` where every specified relative read lies within the
whole domain on the same grid. Offsets are `GridOffset`s such as `ex`, `-ex`,
or `ex/2`. The returned region preserves `r`'s placement and existing selection.
This selects calculation points; it does not read or compute field values.
An empty tuple imposes no extra constraints and returns `r`.
Use this function to choose a fixed region for ordinary, strict indexing.
`@stencil` infers these constraints directly from each update equation.
`interior` instead excludes points on physical boundaries.

```julia
using Tesserae.Stencil: validregion

C = Region(Cell(), (32,24))
ex, ey = unitoffsets(C)
I = validregion(C, (-ex, ex))  # same as validregion(C, -ex, ex)
J = validregion(C, ex)        # right neighbor only
```
"""
function validregion(r::Region{N}, offsets::Tuple{Vararg{GridOffset{N}}}) where {N}
    indices = ntuple(Val(N)) do d
        lo, hi = first(r.indices[d]), last(r.indices[d])
        for offset in offsets
            phase = Int(!isnodealigned(r.location, d))+offset.doubled[d]
            count = r.cells[d]+Int(iseven(phase))
            shift = fld(phase, 2)
            lo = max(lo, 1-shift)
            hi = min(hi, count-shift)
        end
        lo:hi
    end
    Region{N}(r.location, r.cells, indices)
end

validregion(r::Region, e::GridOffset, rest::GridOffset...) = validregion(r, (e, rest...))

"""
    boundaries(r, d)
    boundaries(r)

Intersect `r` with the lower and upper physical boundaries normal to axis `d`,
preserving selections on other axes and retaining singleton dimensions.
The placement must have samples on these boundaries. Omit `d` only when the
placement has exactly one grid-line-aligned axis. A boundary absent from `r`
produces an empty region; it does not become a boundary of the selected subset.
"""
function boundaries(r::Region{N}, d::Int) where {N}
    1 ≤ d ≤ N || throw(ArgumentError("invalid boundary axis $d"))
    isnodealigned(r.location, d) ||
        throw(ArgumentError("this placement has no sampling points on boundaries normal to axis $d"))
    count = r.cells[d]+1
    map((1, count)) do side
        indices = ntuple(j -> j == d ? intersect_range(r.indices[j], side:side) : r.indices[j], Val(N))
        Region{N}(r.location, r.cells, indices)
    end
end

function boundaries(r::Region{N}) where {N}
    count_ones(r.location.mask) == 1 ||
        throw(ArgumentError("specify the boundary axis when it is not unique"))
    boundaries(r, trailing_zeros(r.location.mask)+1)
end

struct ShiftedRegion{N}
    region::Region{N}
    offset::GridOffset{N}
end

@inline Base.:+(r::Region{N}, e::GridOffset{N}) where {N} = ShiftedRegion(r, e)
@inline Base.:+(e::GridOffset{N}, r::Region{N}) where {N} = r+e
@inline Base.:-(r::Region{N}, e::GridOffset{N}) where {N} = r+(-e)
@inline Base.:+(r::ShiftedRegion{N}, e::GridOffset{N}) where {N} = ShiftedRegion(r.region, r.offset+e)
@inline Base.:+(e::GridOffset{N}, r::ShiftedRegion{N}) where {N} = r+e
@inline Base.:-(r::ShiftedRegion{N}, e::GridOffset{N}) where {N} = r+(-e)
Base.broadcastable(r::Union{Region,ShiftedRegion}) = Ref(r)
Base.size(r::ShiftedRegion) = size(r.region)
Base.size(r::ShiftedRegion, d::Integer) = size(r.region, d)
Base.ndims(r::ShiftedRegion) = ndims(r.region)
