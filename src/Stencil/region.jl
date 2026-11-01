@enum AxisRegion::UInt8 _whole _interior _lower _upper

"""
    Region(location, Val(N))

Describe the whole physical domain sampled at `Cell()` or `Face(d)` in `N`
dimensions. A region stores a placement and selectors, without holding an array
or its size. `A[R]` resolves it against `A`; `view(A, R)` returns an ordinary view.
Arrays referring to the same physical domain must have compatible placements
and dimensions. Halos are not part of this API.
"""
struct Region{N}
    location::Location
    selectors::NTuple{N, AxisRegion}
    function Region(location::Location, selectors::NTuple{N, AxisRegion}) where {N}
        checklocation(location, N)
        new{N}(location, selectors)
    end
end

Region(location::Location, ::Val{N}) where {N} =
    Region(location, ntuple(_ -> _whole, Val(N)))

"""
    interior(R)

Keep sampling points strictly inside the physical domain, preserving `R`'s
placement. For a face region this excludes its two boundary faces. For a cell
region it keeps every cell, including cells touching the boundary, because
their centers are inside the domain.
"""
function interior(region::Region{N}) where {N}
    all(s -> s == _whole || s == _interior, region.selectors) ||
        throw(ArgumentError("interior expects a whole or interior region"))
    Region(region.location, ntuple(Val(N)) do d
        isnodealigned(region.location, d) ? _interior : _whole
    end)
end

"""
    boundaries(F, d)

Return the lower and upper boundary face regions along axis `d`. `F` must be
the whole `Face(d)` region. Singleton dimensions are retained when indexing.
"""
function boundaries(region::Region{N}, d::Int) where {N}
    1 <= d <= N || throw(ArgumentError("invalid boundary axis $d"))
    region.location == Face(d) ||
        throw(ArgumentError("boundary faces must be normal to axis $d"))
    all(==(_whole), region.selectors) ||
        throw(ArgumentError("boundaries expects the whole face region"))
    map((_lower, _upper)) do side
        Region(region.location, ntuple(j -> j == d ? side : _whole, Val(N)))
    end
end

struct ShiftedRegion{N}
    region::Region{N}
    offset::GridOffset{N}
end

@inline Base.:+(r::Region{N}, e::GridOffset{N}) where {N} = ShiftedRegion(r, e)
@inline Base.:+(e::GridOffset{N}, r::Region{N}) where {N} = r + e
@inline Base.:-(r::Region{N}, e::GridOffset{N}) where {N} = r + (-e)
@inline Base.:+(r::ShiftedRegion{N}, e::GridOffset{N}) where {N} =
    ShiftedRegion(r.region, r.offset + e)
@inline Base.:+(e::GridOffset{N}, r::ShiftedRegion{N}) where {N} = r + e
@inline Base.:-(r::ShiftedRegion{N}, e::GridOffset{N}) where {N} = r + (-e)

Base.broadcastable(r::Region) = Ref(r)
Base.broadcastable(r::ShiftedRegion) = Ref(r)
