"""
    indexranges(region, array_axes)

Resolve a physical region against concrete array axes and return one storage
index range per dimension.
"""
function indexranges(region::Region{N, <: NTuple{N, Physical}}, array_axes::NTuple{N, AbstractUnitRange{Int}}) where {N}
    ntuple(Val(N)) do d
        axis = array_axes[d]
        halo_width = halo(region)
        node_aligned = isnodealigned(placement(region), d)
        phase = ifelse(node_aligned, 0, 1)

        translated = phase + nhalfsteps(region, d)
        translated_phase = mod(translated, 2)
        index_shift = fld(translated, 2)

        ncells = length(axis) - 2 * halo_width - iszero(translated_phase)
        count = ncells + node_aligned
        first_index = first(axis) + halo_width + index_shift

        first_index:(first_index + count - 1)
    end
end

@inline function Base.to_indices(A::AbstractArray{T, N}, indices::Tuple{Region{N}}) where {T, N}
    indexranges(only(indices), axes(A))
end
