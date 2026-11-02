function placement_name(location::Location, N::Int)
    count = count_ones(location.mask)
    count == 0 && return "Cell()"
    count == 1 && return "Face($(trailing_zeros(location.mask)+1))"
    count == N && return "Vertex()"
    count == N-1 && return "Edge($(findfirst(d -> !isnodealigned(location, d), 1:N)))"
    "grid-line axes $(findall(d -> isnodealigned(location, d), 1:N))"
end

function regionranges(r::Region{N}, e::GridOffset{N}, ax::NTuple{N,AbstractUnitRange{Int}}) where {N}
    ntuple(Val(N)) do d
        source_is_aligned = isnodealigned(r.location, d)
        phase = Int(!source_is_aligned)+e.doubled[d]
        expected = r.cells[d]+iseven(phase)
        if length(ax[d]) != expected
            placement = placement_name(r.location, N)
            shift = map(x -> x/2, e.doubled)
            target = iseven(phase) ? "grid-line" : "cell-center"
            throw(DimensionMismatch(
                "axis $d requires $expected samples at $target positions, got $(length(ax[d])); " *
                "source placement=$placement, shift=$shift cells"))
        end
        lo = first(ax[d])+fld(phase, 2)
        if isempty(r.indices[d])
            first(ax[d]):(first(ax[d])-1)
        else
            indices = sampleindices(r, d)
            (lo+first(indices)-1):(lo+last(indices)-1)
        end
    end
end

@inline function Base.to_indices(A::AbstractArray{V,N}, I::Tuple{Region{N}}) where {V,N}
    regionranges(only(I), zero(GridOffset{N}), axes(A))
end
@inline function Base.to_indices(A::AbstractArray{V,N}, I::Tuple{ShiftedRegion{N}}) where {V,N}
    r = only(I)
    regionranges(r.region, r.offset, axes(A))
end
