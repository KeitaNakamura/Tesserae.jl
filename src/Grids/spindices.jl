# -----------------------------------------------------------------------------
#  Sparse block indices
# -----------------------------------------------------------------------------

struct SpIndex{I}
    index::I
    spindex::Int
end

# Tied to the current numbering of a `SpIndices`, so it is a short-lived token
# and must not be stored across `update_sparsity!` calls.
@inline logicalindex(x::SpIndex) = x.index
@inline storageindex(x::SpIndex) = x.spindex
isactive(x::SpIndex) = !iszero(x.spindex)

@inline function Base.getindex(A::AbstractArray, i::SpIndex)
    @boundscheck checkbounds(A, logicalindex(i))
    @inbounds isactive(i) ? A[logicalindex(i)] : zero_recursive(eltype(A))
end

Base.show(io::IO, x::SpIndex) = print(io, "SpIndex(", x.index, ", ", ifelse(isactive(x), x.spindex, CDot()), ")")

struct ParticleBlockTracker{B <: AbstractArray{Int}, C <: AbstractArray{Int32}}
    blockids::B  # block id currently recorded for each particle
    counts::C    # number of particles currently recorded in each block
end

function ParticleBlockTracker(blocknumbering::AbstractArray)
    blockids = similar(vec(blocknumbering), Int, 0)
    counts = fillzero!(similar(blocknumbering, Int32))
    ParticleBlockTracker(blockids, counts)
end

# Owned by `SpIndices`, so particle-driven sparsity updates reuse storage
# instead of allocating block-sized temporaries every step. The two 1-element
# readback buffers live here for the same reason: on GPU a fresh device
# allocation per step can itself stall the stream.
struct BlockSparsityWorkspace{O <: AbstractArray{Bool}, A <: AbstractArray{Bool}, T <: ParticleBlockTracker,
                              C <: AbstractVector{Int}, G <: AbstractVector{Int32}}
    occupied::O      # blocks containing particles
    active::A        # blocks allocated for basis support
    tracker::T       # particle block ids and per-block particle counts
    active_count::C  # readback of the active block count after renumbering
    changed::G       # readback of the occupied-set change flag
end

function BlockSparsityWorkspace(blocknumbering::AbstractArray)
    occupied = fillzero!(similar(blocknumbering, Bool))
    active = fillzero!(similar(blocknumbering, Bool))
    active_count = similar(vec(blocknumbering), Int, 1)
    changed = fillzero!(similar(vec(blocknumbering), Int32, 1))
    BlockSparsityWorkspace(occupied, active, ParticleBlockTracker(blocknumbering), active_count, changed)
end

# Block sparsity is stored as a dense array over block coordinates.
# Zero means inactive; positive values are compact blocknumbers for SpArray.data.
struct SpIndices{dim, L, B <: AbstractArray{Int, dim}, W <: BlockSparsityWorkspace} <: AbstractArray{SpIndex{CartesianIndex{dim}}, dim}
    dims::Dims{dim}
    blocknumbering::B
    workspace::W
end

function SpIndices(dims::Dims{dim}; block_size_log2::Val{L}=Val(BLOCK_SIZE_LOG2)) where {dim, L}
    _check_block_size_log2(block_size_log2)
    blocknumbering = fill(0, nblocks(dims; block_size_log2))
    workspace = BlockSparsityWorkspace(blocknumbering)
    SpIndices{dim, L, typeof(blocknumbering), typeof(workspace)}(dims, blocknumbering, workspace)
end
SpIndices(dims::Int...; kwargs...) = SpIndices(dims; kwargs...)
SpIndices(mesh::CartesianMesh) = SpIndices(size(mesh); block_size_log2=Val(block_size_log2(mesh)))

Base.size(sp::SpIndices) = sp.dims
Base.IndexStyle(::Type{<: SpIndices}) = IndexCartesian()

@inline blocknumbering(sp::SpIndices) = sp.blocknumbering
@inline sparsity_workspace(sp::SpIndices) = sp.workspace
@inline occupied_blocks(sp::SpIndices) = sparsity_workspace(sp).occupied
@inline active_blocks(sp::SpIndices) = sparsity_workspace(sp).active
@inline sparsity_tracker(sp::SpIndices) = sparsity_workspace(sp).tracker
@inline nblocks(sp::SpIndices) = size(blocknumbering(sp))
@inline block_size_log2(::SpIndices{dim, L}) where {dim, L} = L

# Each active block stores a dense block of size blocksize(sp) in SpArray.data.
@inline blockwidth(sp::SpIndices) = blockwidth(Val(block_size_log2(sp)))
@inline blocksize(sp::SpIndices{dim}) where {dim} = nfill(blockwidth(sp), Val(dim))
@inline blocklength(sp::SpIndices{dim, L}) where {dim, L} = 1 << (L*dim)

# blocknumber + local linear index inside the block -> SpArray.data index.
@inline storageindex(sp::SpIndices, blocknumber::Integer, localindex::Integer) = (blocknumber - 1) * blocklength(sp) + localindex

# Logical node index -> block coordinate.
@inline blockindex(I::Vararg{Integer, dim}; block_size_log2::Val{L}) where {dim, L} =
    @. ((I - 1) >> L) + 1

# Logical node index -> block coordinate and local linear index inside the block.
@inline function global_to_blocklocal(I::Vararg{Integer, dim}; block_size_log2::Val{L}) where {dim, L}
    j = I .- 1
    block = blockindex(I...; block_size_log2)
    localcoord = @. (j & ((1 << L) - 1)) + 1
    LI = LinearIndices(nfill(1 << L, Val(dim)))
    @inbounds block, LI[localcoord...]
end

# block coordinate and local Cartesian index inside the block -> logical node index.
@inline function blocklocal_to_global(block::CartesianIndex{dim}, localcoord::CartesianIndex{dim}; block_size_log2::Val{L}) where {dim, L}
    CartesianIndex(ntuple(d -> ((block[d] - 1) << L) + localcoord[d], Val(dim)))
end

# GPU kernels cannot iterate `activeindices(spinds)` directly, so they launch over
# block-local slots and recover the corresponding active `SpIndex` with these helpers.
@inline _spindex_ndrange(spinds::SpIndices) = length(blocknumbering(spinds)) * blocklength(spinds)

@inline function _active_spindex(spinds::SpIndices, blocknumber, block::CartesianIndex, l::Integer, localindices)
    iszero(blocknumber) && return false, SpIndex(block, 0)
    @inbounds localcoord = localindices[l]
    I = blocklocal_to_global(block, localcoord; block_size_log2=Val(block_size_log2(spinds)))
    checkbounds(Bool, spinds, Tuple(I)...) || return false, SpIndex(I, 0)
    true, SpIndex(I, storageindex(spinds, blocknumber, l))
end

@inline function _active_spindex(spinds::SpIndices, k::Integer)
    numbering = blocknumbering(spinds)
    blocks = CartesianIndices(numbering)
    localindices = CartesianIndices(blocksize(spinds))
    nlocal = length(localindices)
    b = (k - 1) ÷ nlocal + 1
    l = (k - 1) % nlocal + 1
    @inbounds blocknumber = numbering[b]
    @inbounds block = blocks[b]
    _active_spindex(spinds, blocknumber, block, l, localindices)
end

@inline function Base.getindex(sp::SpIndices{dim}, I::Vararg{Integer, dim}) where {dim}
    @boundscheck checkbounds(sp, I...)
    block_size = Val(block_size_log2(sp))
    block, localindex = global_to_blocklocal(I...; block_size_log2=block_size)
    @inbounds blocknumber = blocknumbering(sp)[block...]
    index = storageindex(sp, blocknumber, localindex)
    SpIndex(CartesianIndex(I), ifelse(iszero(blocknumber), zero(index), index))
end

struct ActiveSpIndices{dim, S <: SpIndices{dim}}
    spinds::S
end

# Storage order, deliberately not Cartesian order, so callers working with
# `SpArray.data` can use the resulting `SpIndex` values without re-sorting.
activeindices(sp::SpIndices) = ActiveSpIndices(sp)

Base.IteratorSize(::Type{<: ActiveSpIndices}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<: ActiveSpIndices}) = Base.HasEltype()
Base.eltype(::Type{ActiveSpIndices{dim, S}}) where {dim, S} = SpIndex{CartesianIndex{dim}}

function Base.iterate(iter::ActiveSpIndices{dim}, state=(1, 1)) where {dim}
    sp = iter.spinds
    numbering = blocknumbering(sp)
    blocks = CartesianIndices(numbering)
    localindices = CartesianIndices(blocksize(sp))
    nblock = length(numbering)
    nlocal = length(localindices)
    b, l = state

    @inbounds while b ≤ nblock
        blocknumber = numbering[b]
        if !iszero(blocknumber)
            block = blocks[b]
            while l ≤ nlocal
                active, spindex = _active_spindex(sp, blocknumber, block, l, localindices)
                l += 1
                active && return spindex, (b, l)
            end
        end
        b += 1
        l = 1
    end

    nothing
end

@inline function isactive(sp::SpIndices{dim}, I::Vararg{Integer, dim}) where {dim}
    @boundscheck checkbounds(sp, I...)
    block = blockindex(I...; block_size_log2=Val(block_size_log2(sp)))
    @inbounds !iszero(blocknumbering(sp)[block...])
end
@inline isactive(sp::SpIndices, I::CartesianIndex) = (@_propagate_inbounds_meta; isactive(sp, Tuple(I)...))
