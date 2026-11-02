"""
    coordinates(r, h; origin=zero.(h))

Iterate over coordinate tuples at the sampling points selected by `r`.
`h` contains the positive spacing along each axis; `origin` is the lower
physical corner of the whole domain, including when `r` selects a boundary
or interior subset. Shifted regions translate the selected sampling points.

The returned standard Julia iterator has shape `size(r)` and does not
materialize a coordinate array. Comprehensions preserve this shape, including
singleton boundary dimensions and empty selections.

```julia
C = Region(Cell(), (32,24))
h = (1/32, 1/24)
temperature = [prod(sinpi, x) for x in coordinates(C, h)]
```
"""
function coordinates(r::Union{Region,ShiftedRegion}, h; origin = zero.(h))
    N = ndims(r)
    length(h) == N ||
        throw(DimensionMismatch("expected $N grid spacings, got $(length(h))"))
    length(origin) == N ||
        throw(DimensionMismatch("expected $N origin coordinates, got $(length(origin))"))
    all(hd -> hd isa Real && isfinite(hd) && hd > 0, h) ||
        throw(ArgumentError("grid spacings must be finite positive real numbers"))
    all(x -> x isa Real && isfinite(x), origin) ||
        throw(ArgumentError("origin coordinates must be finite real numbers"))
    source = r isa Region ? r : r.region
    offset = r isa Region ? zero(GridOffset{N}) : r.offset
    coordinate_axes = ntuple(Val(N)) do d
        hd, x0 = h[d], origin[d]
        on_grid = isnodealigned(source.location, d)
        count = source.cells[d]+Int(on_grid)
        phase = on_grid ? 0 : one(hd)/2
        axis = ((0:(count-1)) .+ phase) .* hd .+ (x0+offset.doubled[d]*(hd/2))
        axis[sampleindices(source, d)]
    end
    Iterators.product(coordinate_axes...)
end
