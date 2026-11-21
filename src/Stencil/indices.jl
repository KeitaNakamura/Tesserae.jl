function placement_name(location::Location, N::Int)
    count = count_ones(location.mask)
    count == 0 && return "Cell()"
    count == 1 && return "Face($(trailing_zeros(location.mask)+1))"
    count == N && return "Vertex()"
    count == N-1 && return "Edge($(findfirst(d -> !isnodealigned(location, d), 1:N)))"
    "grid-line axes $(findall(d -> isnodealigned(location, d), 1:N))"
end

function regionranges(r::Region{N}, ax::NTuple{N, AbstractUnitRange{Int}}) where {N}
    counts = samplecounts(r.location, r.cells)
    ntuple(Val(N)) do d
        expected = counts[d]
        origin = first(ax[d])
        if length(ax[d]) != expected && 1-first(ax[d]) == last(ax[d])-expected && first(ax[d]) < 1
            origin = 1 # Symmetric logical halo around physical indices 1:expected.
        elseif length(ax[d]) != expected
            placement = placement_name(r.location, N)
            throw(DimensionMismatch("axis $d requires $expected samples at $placement, got $(length(ax[d]))"))
        end
        r.indices[d] .+ (origin-1)
    end
end

@inline function Base.to_indices(A::AbstractArray{V, N}, I::Tuple{Region{N}}) where {V, N}
    regionranges(only(I), axes(A))
end

"""
    regionindices(A, R)

Return the tuple of array index ranges used by `A[R]` and `view(A, R)`.
The ranges use `A`'s axes and preserve the region's order, including after
reflection. The array must have the sample counts required by `R`'s placement,
optionally with symmetric logical halos. A shape mismatch raises
`DimensionMismatch`.

This computes indices without reading field values or clipping to storage.
A shifted region can therefore return out-of-bounds ranges; accessing them
with `view` or indexing performs the normal bounds checks. `@stencil` applies
its separate common-range rule to all explicit reads and writes.
"""
@inline function regionindices(A::AbstractArray{T, N}, R::Region{N}) where {T, N}
    Base.to_indices(A, (R,))
end

# Resolve logical indices before calling a backend's own view implementation.
function native_view(A::AbstractArray, indices::Tuple)
    ranges = map(indices) do index
        index isa AbstractRange && step(index) == 1 ? (first(index):last(index)) : index
    end
    view(A, ranges...)
end
function native_view(A::OffsetArray, indices::Tuple)
    shifted = map(indices, axes(A), axes(parent(A))) do index, logical, storage
        index .+ (first(storage)-first(logical))
    end
    native_view(parent(A), shifted)
end

function Base.view(A::OffsetArray, r::Region)
    native_view(A, Base.to_indices(A, (r,)))
end

# Parallel planes form one ordinary Cartesian selection, like A[[1,end],:,:].
# A step range keeps the selector small and usable by native GPU broadcasts.
function Base.to_indices(A::AbstractArray{T, N}, I::Tuple{<:AbstractVector{<:Region{N}}}) where {T, N}
    regions = only(I)
    1 ≤ length(regions) ≤ 2 || throw(ArgumentError("select one or two parallel Region planes"))
    a = first(regions)
    left = Base.to_indices(A, (a,))
    length(regions) == 1 && return left
    b = last(regions)
    checkframe(a.frame, b.frame)
    a.cells == b.cells && a.location == b.location || throw(ArgumentError("combined Regions must share a domain and placement"))
    right = Base.to_indices(A, (b,))
    any(isempty, left) && return right
    any(isempty, right) && return left
    axis = findfirst(d -> left[d] != right[d], 1:N)
    axis === nothing && return left
    all(d -> d == axis || left[d] == right[d], 1:N) ||
        throw(ArgumentError("combined Regions must be parallel planes with matching transverse ranges"))
    length(left[axis]) == length(right[axis]) == 1 || throw(ArgumentError("each selected Region must be one plane along the combined axis"))
    lo, hi = first(left[axis]), first(right[axis])
    distance = hi-lo
    ntuple(d -> d==axis ? (lo:distance:hi) : left[d], Val(N))
end

function Base.view(A::OffsetArray, regions::AbstractVector{<:Region})
    native_view(A, Base.to_indices(A, (regions,)))
end
