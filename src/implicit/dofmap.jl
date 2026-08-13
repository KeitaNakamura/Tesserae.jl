# -----------------------------------------------------------------------------
#  DofMap
# -----------------------------------------------------------------------------

abstract type AbstractDofMap end

# ---- DofMap ----

"""
    DofMap(mask::AbstractArray{Bool})

Create a degree of freedom (DoF) map from a `mask` of size `(ndofs, size(grid)...)`.
`ndofs` represents the number of DoFs stored at each grid location.

```jldoctest
julia> mesh = CartesianMesh(1, (0,2), (0,1));

julia> grid = generate_grid(@NamedTuple{x::Vec{2, Float64}, v::Vec{2, Float64}}, mesh);

julia> grid.v .= reshape(reinterpret(Vec{2, Float64}, 1.0:12.0), 3, 2)
3×2 Matrix{Vec{2, Float64}}:
 [1.0, 2.0]  [7.0, 8.0]
 [3.0, 4.0]  [9.0, 10.0]
 [5.0, 6.0]  [11.0, 12.0]

julia> dofmask = falses(2, size(grid)...);

julia> dofmask[1,1:2,:] .= true; # activate nodes

julia> dofmask[:,3,2] .= true; # activate nodes

julia> reinterpret(reshape, Vec{2, Bool}, dofmask)
3×2 reinterpret(reshape, Vec{2, Bool}, ::BitArray{3}) with eltype Vec{2, Bool}:
 [1, 0]  [1, 0]
 [1, 0]  [1, 0]
 [0, 0]  [1, 1]

julia> free = DofMap(dofmask);

julia> free(grid.v)
6-element view(reinterpret(reshape, Float64, ::Matrix{Vec{2, Float64}}), CartesianIndex{3}[CartesianIndex(1, 1, 1), CartesianIndex(1, 2, 1), CartesianIndex(1, 1, 2), CartesianIndex(1, 2, 2), CartesianIndex(1, 3, 2), CartesianIndex(2, 3, 2)]) with eltype Float64:
  1.0
  3.0
  7.0
  9.0
 11.0
 12.0
```
"""
struct DofMap{N, I <: AbstractVector{<: CartesianIndex}, J <: AbstractVector{<: CartesianIndex}} <: AbstractDofMap
    masksize::Dims{N}
    indices::I # (dof, x, y, z)
    indices4scalar::J # (dof, x, y, z)
end

# ---- BlockDofMap ----

"""
    BlockDofMap(masks::Tuple)

Create a block-major DoF map from one Boolean mask per block. The block order
must match the order passed to `create_block_sparse_matrix`. Indexing the map
returns the `DofMap` for one block.

```julia
free = BlockDofMap((velocity_mask, pressure_mask))
A = extract(blocks, free)
Aup = extract(blocks[1,2], free[1], free[2])
```
"""
struct BlockDofMap{M <: Tuple{Vararg{DofMap}}} <: AbstractDofMap
    maps::M
    indices::Vector{Int}
end

# ---- construction ----

function DofMap(mask::AbstractArray{Bool})
    masksize = size(mask)
    I = findall(mask)
    J = map(i -> CartesianIndex(1, Base.tail(Tuple(i))...), I)
    DofMap(masksize, I, J)
end

function BlockDofMap(masks::Tuple{Vararg{AbstractArray{Bool}}})
    isempty(masks) && throw(ArgumentError("at least one block mask is required"))
    maps = map(DofMap, masks)
    indices = Int[]
    sizehint!(indices, sum(ndofs, maps))
    offset = 0
    for dofmap in maps
        linear_indices = LinearIndices(dofmap.masksize)
        for index in dofmap.indices
            push!(indices, offset + linear_indices[index])
        end
        offset += length(linear_indices)
    end
    BlockDofMap(maps, indices)
end

"""
    dofmap(mask::AbstractArray{Bool})

Create a `DofMap` from one Boolean mask.
"""
dofmap(mask::AbstractArray{Bool}) = DofMap(mask)

"""
    dofmap(masks::Tuple)

Create a `BlockDofMap` from a tuple containing one Boolean mask per block.
"""
dofmap(masks::Tuple{Vararg{AbstractArray{Bool}}}) = BlockDofMap(masks)

# ---- indexing ----

function (dofmap::DofMap)(A::AbstractArray{T}) where {T <: Vec{1}}
    A′ = reshape(reinterpret(eltype(T), A), 1, size(A)...)
    @boundscheck checkbounds(A′, dofmap.indices)
    @inbounds view(A′, dofmap.indices)
end
function (dofmap::DofMap)(A::AbstractArray{T}) where {T <: Vec}
    A′ = reinterpret(reshape, eltype(T), A)
    @boundscheck checkbounds(A′, dofmap.indices)
    @inbounds view(A′, dofmap.indices)
end

function (dofmap::DofMap)(A::AbstractArray{T}) where {T <: Real}
    A′ = reshape(A, 1, size(A)...)
    @boundscheck checkbounds(A′, dofmap.indices4scalar)
    @inbounds view(A′, dofmap.indices4scalar)
end

Base.length(dofmap::BlockDofMap) = length(dofmap.maps)
Base.getindex(dofmap::BlockDofMap, i::Int) = dofmap.maps[i]

ndofs(dofmap::DofMap) = length(dofmap.indices)
ndofs(dofmap::BlockDofMap) = length(dofmap.indices)
dofs(dofmap::DofMap) = LinearIndices(dofmap.masksize)[dofmap.indices]
dofs(dofmap::BlockDofMap) = dofmap.indices

full_ndofs(dofmap::DofMap) = prod(dofmap.masksize)
full_ndofs(dofmap::BlockDofMap) = sum(full_ndofs, dofmap.maps)
dofs(colon::Colon) = colon

