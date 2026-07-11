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

        ncells = length(axis) - 2 * width - iszero(translated_phase)
        count = ncells + node_aligned
        first_index = first(axis) + width + index_shift
        physical = first_index:(first_index + count - 1)

        _indexrange(axisregion(region, d), physical, width, node_aligned)
    end
end

@inline _indexrange(::Physical, physical::UnitRange{Int}, ::Int, ::Bool) = physical

@inline function _indexrange(region::Halo, physical::UnitRange{Int}, width::Int, ::Bool)
    s = side(region)
    s == -1 && return (first(physical) - width):(first(physical) - 1)
    s == +1 && return (last(physical) + 1):(last(physical) + width)
    throw(ArgumentError("side must be -1 or +1, got $s"))
end

@inline function _indexrange(region::Boundary, physical::UnitRange{Int}, ::Int, ::Bool)
    s = side(region)
    s == -1 && return first(physical):first(physical)
    s == +1 && return last(physical):last(physical)
    throw(ArgumentError("side must be -1 or +1, got $s"))
end

@inline function Base.to_indices(A::AbstractArray{T, N}, indices::Tuple{Region{N}}) where {T, N}
    indexranges(only(indices), axes(A))
end

@inline function Base.to_indices(A::AbstractArray, indices::Tuple{<: RegionMap})
    mappedranges(only(indices), axes(A))
end
