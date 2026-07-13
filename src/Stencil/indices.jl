"""
    regionranges(region, array_axes)

Resolve a region against concrete array axes and return one storage
index range per dimension.
"""
function regionranges(region::Region{N}, array_axes::NTuple{N, AbstractUnitRange{Int}}) where {N}
    regionranges(region, zero(GridOffset{N}), array_axes)
end

function regionranges(shifted::ShiftedRegion{N}, array_axes::NTuple{N, AbstractUnitRange{Int}}) where {N}
    regionranges(parent(shifted), offset(shifted), array_axes)
end

function regionranges(region::Region{N}, offset::GridOffset{N}, array_axes::NTuple{N, AbstractUnitRange{Int}}) where {N}
    dimensions = ntuple(identity, Val(N))

    map(axisregions(region), array_axes, halowidth(region), dimensions) do region_axis, array_axis, halowidth, d
        node_aligned = isnodealigned(location(region), d)
        phase = ifelse(node_aligned, 0, 1)

        translated = phase + nhalfsteps(offset, d)
        translated_phase = mod(translated, 2)
        index_shift = fld(translated, 2)

        available = length(array_axis) - iszero(translated_phase)
        available ≥ 1 && halowidth ≤ (available - 1) ÷ 2 || throw(DimensionMismatch("axis $d with length $(length(array_axis)) does not contain a physical cell with halowidth $halowidth"))
        ncells = available - 2 * halowidth
        count = ncells + node_aligned
        first_index = first(array_axis) + halowidth + index_shift
        physical_range = first_index:(first_index + count - 1)

        axisrange(region_axis, array_axis, physical_range, halowidth)
    end
end

@inline function axisrange(region::AxisRegion, array_axis::AbstractUnitRange{Int}, physical_range::UnitRange{Int}, halowidth::Int)
    if isfull(region)
        first(array_axis):last(array_axis)
    elseif isphysical(region)
        physical_range
    elseif ishalo(region) && side(region) == -1
        (first(physical_range) - halowidth):(first(physical_range) - 1)
    elseif ishalo(region)
        (last(physical_range) + 1):(last(physical_range) + halowidth)
    elseif side(region) == -1
        first(physical_range):first(physical_range)
    else
        last(physical_range):last(physical_range)
    end
end

@inline function Base.to_indices(A::AbstractArray{T, N}, indices::Tuple{Region{N}}) where {T, N}
    regionranges(only(indices), axes(A))
end

@inline function Base.to_indices(A::AbstractArray{T, N}, indices::Tuple{ShiftedRegion{N}}) where {T, N}
    regionranges(only(indices), axes(A))
end

@inline function Base.to_indices(A::AbstractArray, indices::Tuple{<: RegionMap})
    mappedranges(only(indices), axes(A))
end
