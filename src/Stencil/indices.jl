"""
    indexranges(region, array_axes)

Resolve a region against concrete array axes and return one storage
index range per dimension.
"""
function indexranges(region::Region{N}, array_axes::NTuple{N, AbstractUnitRange{Int}}) where {N}
    ntuple(Val(N)) do d
        axis = array_axes[d]
        width = halowidth(region, d)
        node_aligned = isnodealigned(placement(region), d)
        phase = ifelse(node_aligned, 0, 1)

        translated = phase + nhalfsteps(region, d)
        translated_phase = mod(translated, 2)
        index_shift = fld(translated, 2)

        available = length(axis) - iszero(translated_phase)
        available ≥ 1 && width ≤ (available - 1) ÷ 2 || throw(DimensionMismatch("axis $d with length $(length(axis)) does not contain a physical cell with halowidth $width"))
        ncells = available - 2 * width
        count = ncells + node_aligned
        first_index = first(axis) + width + index_shift
        physical = first_index:(first_index + count - 1)

        _indexrange(axisregion(region, d), physical, width)
    end
end

@inline _indexrange(::Physical, physical::UnitRange{Int}, ::Int) = physical

@inline function _indexrange(region::Halo, physical::UnitRange{Int}, width::Int)
    if side(region) == -1
        (first(physical) - width):(first(physical) - 1)
    else
        (last(physical) + 1):(last(physical) + width)
    end
end

@inline function _indexrange(region::Boundary, physical::UnitRange{Int}, ::Int)
    if side(region) == -1
        first(physical):first(physical)
    else
        last(physical):last(physical)
    end
end

@inline function Base.to_indices(A::AbstractArray{T, N}, indices::Tuple{Region{N}}) where {T, N}
    indexranges(only(indices), axes(A))
end

@inline function Base.to_indices(A::AbstractArray, indices::Tuple{<: RegionMap})
    mappedranges(only(indices), axes(A))
end
