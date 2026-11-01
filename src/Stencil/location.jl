"""Sampling placement on a Cartesian grid; set bits denote face-aligned axes."""
struct Location
    mask::UInt
end

"""`Cell()` places a quantity at cell centers."""
Cell() = Location(zero(UInt))

"""`Face(d)` places a quantity at the centers of faces normal to axis `d`."""
function Face(d::Int)
    1 <= d <= 8sizeof(UInt) || throw(ArgumentError("invalid face axis $d"))
    Location(one(UInt) << (d - 1))
end

@inline isnodealigned(location::Location, d::Int) =
    !iszero(location.mask & (one(UInt) << (d - 1)))

function checklocation(location::Location, N::Int)
    1 <= N <= 8sizeof(UInt) || throw(ArgumentError("unsupported dimension $N"))
    (location == Cell() || any(d -> location == Face(d), 1:N)) ||
        throw(ArgumentError("placement is incompatible with $N dimensions"))
    nothing
end

"""
    similar(A, source, target)

Allocate an ordinary array with the same element type and backend as `A`,
changing its sampling placement from `source` to `target`. For example,
`similar(T, Cell(), Face(1))` has one more sample along axis 1 than `T`.
"""
function Base.similar(A::AbstractArray, source::Location, target::Location)
    checklocation(source, ndims(A))
    checklocation(target, ndims(A))
    n = ntuple(ndims(A)) do d
        size(A, d) + isnodealigned(target, d) - isnodealigned(source, d)
    end
    all(>(0), n) || throw(DimensionMismatch("target placement has no samples"))
    similar(A, eltype(A), n)
end
