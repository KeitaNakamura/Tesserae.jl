"""
    coordinates(r)
    coordinates(r, h; origin)

Iterate over coordinate tuples at the ordered sample positions of `r`, without
materializing a coordinate array. Mesh-bound regions carry their origin and
spacing. Logical regions accept explicit spacings and default to zero origin.
Singleton dimensions, empty selections and reflected order are preserved.
"""
function coordinates(r::Region, h; origin=isnothing(r.frame.axes) ? map(zero, h) : r.frame.origin)
    N = ndims(r)
    length(h) == N || throw(DimensionMismatch("expected $N grid spacings, got $(length(h))"))
    length(origin) == N || throw(DimensionMismatch("expected $N origin coordinates, got $(length(origin))"))
    all(hd -> hd isa Real && isfinite(hd) && hd > 0, h) ||
        throw(ArgumentError("grid spacings must be finite positive real numbers"))
    all(x -> x isa Real && isfinite(x), origin) || throw(ArgumentError("origin coordinates must be finite real numbers"))
    if !isnothing(r.frame.axes)
        Tuple(origin) == r.frame.origin && all(==(r.frame.spacing), h) ||
            throw(ArgumentError("coordinates must use the region's grid frame"))
        return coordinates(r)
    end
    coordinate_axes = ntuple(Val(N)) do d
        coordinate_axis(r.indices[d], isnodealigned(r.location, d), h[d], origin[d])
    end
    Iterators.product(coordinate_axes...)
end

function coordinates(r::Region{N}) where {N}
    isfinite(r.frame.spacing) && r.frame.spacing > 0 ||
        throw(ArgumentError("grid spacings must be finite positive real numbers"))
    all(isfinite, r.frame.origin) || throw(ArgumentError("origin coordinates must be finite real numbers"))
    coordinate_axes = ntuple(Val(N)) do d
        coordinate_axis(r.indices[d], isnodealigned(r.location, d), r.frame, d)
    end
    Iterators.product(coordinate_axes...)
end
