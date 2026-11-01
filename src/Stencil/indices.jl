function regionranges(region::Region{N}, offset::GridOffset{N}, array_axes::NTuple{N, AbstractUnitRange{Int}}) where {N}
    ntuple(Val(N)) do d
        face_aligned = isnodealigned(region.location, d)
        phase = Int(!face_aligned) + offset.doubled[d]
        target_is_face = iseven(phase)
        ncells = length(array_axes[d]) - target_is_face
        ncells >= 1 || throw(DimensionMismatch("axis $d contains no physical cell"))

        # The region retains its extent as it moves between placements. A
        # boundary face shifted inward therefore selects its one adjacent cell.
        lo = first(array_axes[d]) + fld(phase, 2)
        hi = lo + ncells + face_aligned - 1
        selector = region.selectors[d]
        if selector == _interior
            (lo + Int(face_aligned)):(hi - Int(face_aligned))
        elseif selector == _lower
            lo:lo
        elseif selector == _upper
            hi:hi
        else
            lo:hi
        end
    end
end

@inline function Base.to_indices(A::AbstractArray{T, N}, I::Tuple{Region{N}}) where {T, N}
    regionranges(only(I), zero(GridOffset{N}), axes(A))
end

@inline function Base.to_indices(A::AbstractArray{T, N}, I::Tuple{ShiftedRegion{N}}) where {T, N}
    r = only(I)
    regionranges(r.region, r.offset, axes(A))
end
