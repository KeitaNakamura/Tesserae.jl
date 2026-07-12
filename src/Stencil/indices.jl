"""
    indexranges(region, array_axes)

Resolve a region against concrete array axes and return one storage
index range per dimension.
"""
function indexranges(region::Region{N}, array_axes::NTuple{N, AbstractUnitRange{Int}}) where {N}
    _indexranges(region, zero(GridOffset{N}), array_axes)
end

function indexranges(shifted::ShiftedRegion{N}, array_axes::NTuple{N, AbstractUnitRange{Int}}) where {N}
    _indexranges(parent(shifted), offset(shifted), array_axes)
end

function _indexranges(region::Region{N}, offset::GridOffset{N}, array_axes::NTuple{N, AbstractUnitRange{Int}}) where {N}
    dimensions = ntuple(identity, Val(N))

    map(axisregions(region), array_axes, halowidth(region), dimensions) do region_axis, array_axis, width, d
        node_aligned = isnodealigned(placement(region), d)
        phase = ifelse(node_aligned, 0, 1)

        translated = phase + nhalfsteps(offset, d)
        translated_phase = mod(translated, 2)
        index_shift = fld(translated, 2)

        available = length(array_axis) - iszero(translated_phase)
        available ≥ 1 && width ≤ (available - 1) ÷ 2 || throw(DimensionMismatch("axis $d with length $(length(array_axis)) does not contain a physical cell with halowidth $width"))
        ncells = available - 2 * width
        count = ncells + node_aligned
        first_index = first(array_axis) + width + index_shift
        physical = first_index:(first_index + count - 1)

        _indexrange(region_axis, physical, width, array_axis)
    end
end

@inline _indexrange(::Full, ::UnitRange{Int}, ::Int, array_axis::AbstractUnitRange{Int}) = first(array_axis):last(array_axis)

@inline _indexrange(::Physical, physical::UnitRange{Int}, ::Int, ::AbstractUnitRange{Int}) = physical

@inline function _indexrange(region::Halo, physical::UnitRange{Int}, width::Int, ::AbstractUnitRange{Int})
    if side(region) == -1
        (first(physical) - width):(first(physical) - 1)
    else
        (last(physical) + 1):(last(physical) + width)
    end
end

@inline function _indexrange(region::Boundary, physical::UnitRange{Int}, ::Int, ::AbstractUnitRange{Int})
    if side(region) == -1
        first(physical):first(physical)
    else
        last(physical):last(physical)
    end
end

@inline function Base.to_indices(A::AbstractArray{T, N}, indices::Tuple{Region{N}}) where {T, N}
    indexranges(only(indices), axes(A))
end

@inline function Base.to_indices(A::AbstractArray{T, N}, indices::Tuple{ShiftedRegion{N}}) where {T, N}
    indexranges(only(indices), axes(A))
end

@inline function Base.to_indices(A::AbstractArray, indices::Tuple{<: RegionMap})
    mappedranges(only(indices), axes(A))
end
