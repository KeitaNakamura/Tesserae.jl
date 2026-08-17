# -----------------------------------------------------------------------------
#  Sparsity updates
# -----------------------------------------------------------------------------

@inline elone(A) = one(eltype(A))
@inline elzero(A) = zero(eltype(A))

function _check_nblocks(sp::SpIndices, blocks)
    nblocks(sp) == size(blocks) || throw(ArgumentError("blocks per dimension $(nblocks(sp)) must match"))
end
function _check_nblocks(sp::SpIndices, mesh::CartesianMesh)
    nblocks(sp) == nblocks(mesh) || throw(ArgumentError("blocks per dimension $(nblocks(sp)) must match"))
end

function _check_same_backend(label, x, backend)
    get_backend(x) == backend || throw(ArgumentError("SpIndices and $label must live on the same backend"))
end

function update_sparsity!(sp::SpIndices, blkspy::AbstractArray)
    _apply_block_activity!(sp, blkspy)
end

function update_sparsity!(spinds::SpIndices, xₚ::AbstractVector{<: Vec}, mesh::CartesianMesh)
    backend = get_backend(blocknumbering(spinds))
    _check_same_backend("particle positions", xₚ, backend)
    _check_same_backend("mesh", mesh, backend)
    _check_nblocks(spinds, mesh)

    if isempty(xₚ)
        _reset_particle_block_tracker!(spinds, 0)
        return _apply_block_activity!(spinds, fillzero!(active_blocks(spinds)); preserve_tracker=true)
    end

    # Reuse numbering when the occupied block set is unchanged.
    if !_update_particle_block_tracker!(spinds, xₚ, mesh, backend)
        return nothing
    end
    activity = _activate_neighbor_blocks!(active_blocks(spinds), occupied_blocks(spinds), backend)
    _apply_block_activity!(spinds, activity; preserve_tracker=true)
end

# Block activity -> compact block numbering.
function _apply_block_activity!(sp::SpIndices, activity; preserve_tracker::Bool=false)
    _check_nblocks(sp, activity)
    _check_same_backend("block activity", activity, get_backend(blocknumbering(sp)))
    n = _number_blocks!(sp, activity)
    preserve_tracker || _invalidate_particle_block_tracker!(sp)
    n
end

_number_blocks!(sp::SpIndices, activity) = _number_blocks!(sp, activity, get_backend(blocknumbering(sp)))

function _number_blocks!(sp::SpIndices, activity, ::CPU)
    numbers = blocknumbering(sp)
    active_block_count = 0
    @inbounds for i in eachindex(numbers, activity)
        numbers[i] = iszero(activity[i]) ? 0 : (active_block_count += 1)
    end
    active_block_count * blocklength(sp)
end

@kernel function gpukernel_init_block_numbering!(block_numbers, @Const(activity))
    b = @index(Global)
    @inbounds block_numbers[b] = ifelse(iszero(activity[b]), elzero(block_numbers), elone(block_numbers))
end

@kernel function gpukernel_finalize_block_numbering!(block_numbers, @Const(activity), active_count)
    b = @index(Global)
    @inbounds if b == length(block_numbers)
        active_count[] = block_numbers[b]
    end
    @inbounds if iszero(activity[b])
        block_numbers[b] = 0
    end
end

function _number_blocks!(sp::SpIndices, activity, backend::GPU)
    block_numbers = blocknumbering(sp)
    active_count_buffer = sparsity_workspace(sp).active_count

    # Build compact block numbers with an inclusive scan:
    # activity -> 0/1 markers -> prefix sum -> inactive blocks reset to 0.
    init_kernel = gpukernel_init_block_numbering!(backend)
    init_kernel(block_numbers, activity; ndrange=length(block_numbers))

    cumsum!(vec(block_numbers), vec(block_numbers))

    finalize_kernel = gpukernel_finalize_block_numbering!(backend)
    finalize_kernel(block_numbers, activity, active_count_buffer; ndrange=length(block_numbers))
    # Only sync before the CPU reads `active_count_buffer`.
    synchronize(backend)
    only(Array(active_count_buffer)) * blocklength(sp)
end

# Manual sparsity updates bypass particle positions, so the tracker no longer
# describes the current sparsity state.
function _invalidate_particle_block_tracker!(spinds::SpIndices)
    tracker = sparsity_tracker(spinds)
    resize_fillzero!(tracker.blockids, 0)
    fillzero!(tracker.counts)
    fillzero!(occupied_blocks(spinds))
    spinds
end

# Rebuild tracker storage on the first particle update, after invalidation, or
# when the particle count changes.
function _reset_particle_block_tracker!(spinds::SpIndices, nparticles::Integer)
    tracker = sparsity_tracker(spinds)
    length(tracker.blockids) == nparticles && return false
    resize_fillzero!(tracker.blockids, nparticles)
    fillzero!(tracker.counts)
    fillzero!(occupied_blocks(spinds))
    true
end

@inline blockid(dims::Dims, x::Vec, mesh::CartesianMesh)::Int = sub2ind(dims, findblock(x, mesh))

# The expensive active expansion and numbering are needed only when the occupied
# blocks change, which is what the return value reports.
function _update_particle_block_tracker!(spinds::SpIndices, xₚ, mesh, ::CPU)
    reset = _reset_particle_block_tracker!(spinds, length(xₚ))
    tracker = sparsity_tracker(spinds)
    blockids = tracker.blockids
    counts = tracker.counts
    moved = reset

    @inbounds for p in eachindex(xₚ)
        new = blockid(size(counts), xₚ[p], mesh)
        old = blockids[p]
        if old != new
            blockids[p] = new
            moved = true
            iszero(old) || (counts[old] -= elone(counts))
            iszero(new) || (counts[new] += elone(counts))
        end
    end

    moved && _refresh_occupied_blocks!(occupied_blocks(spinds), counts, reset)
end

function _refresh_occupied_blocks!(occupied, counts, tracker_reset::Bool)
    changed = tracker_reset
    @inbounds for i in eachindex(occupied, counts)
        now = !iszero(counts[i])
        if now != !iszero(occupied[i])
            occupied[i] = now
            changed = true
        end
    end
    changed
end

@kernel function gpukernel_update_particle_block_tracker!(blockids, counts, @Const(xₚ), @Const(mesh))
    p = @index(Global)
    new = blockid(size(counts), xₚ[p], mesh)
    @inbounds old = blockids[p]
    if old != new
        @inbounds blockids[p] = new
        if !iszero(old)
            @inbounds Atomix.@atomic counts[old] -= elone(counts)
        end
        if !iszero(new)
            @inbounds Atomix.@atomic counts[new] += elone(counts)
        end
    end
end

@kernel function gpukernel_refresh_occupied_blocks!(occupied_blocks, @Const(counts), changed)
    b = @index(Global)
    @inbounds begin
        now = !iszero(counts[b])
        if now != !iszero(occupied_blocks[b])
            occupied_blocks[b] = now
            Atomix.@atomic changed[] += elone(changed)
        end
    end
end

function _update_particle_block_tracker!(spinds::SpIndices, xₚ, mesh, backend::GPU)
    reset = _reset_particle_block_tracker!(spinds, length(xₚ))
    tracker = sparsity_tracker(spinds)
    # Only the occupied set decides whether expansion and numbering are needed;
    # individual particle moves are an intermediate detail.
    changed = fillzero!(sparsity_workspace(spinds).changed)

    update_kernel = gpukernel_update_particle_block_tracker!(backend)
    update_kernel(tracker.blockids, tracker.counts, xₚ, mesh; ndrange=length(xₚ))

    # Launches on the same backend are ordered; only sync before the CPU reads `changed`.
    refresh_kernel = gpukernel_refresh_occupied_blocks!(backend)
    refresh_kernel(occupied_blocks(spinds), tracker.counts, changed; ndrange=length(occupied_blocks(spinds)))
    synchronize(backend)
    reset || !iszero(only(Array(changed)))
end

# Occupied blocks -> active blocks for basis support.
function _activate_block_neighborhood!(active_blocks, I::CartesianIndex, CI)
    blks = (I - oneunit(I)):(I + oneunit(I))
    active_blocks[blks ∩ CI] .= true
    active_blocks
end

function _activate_neighbor_blocks!(active, occupied, ::CPU)
    fillzero!(active)
    CI = CartesianIndices(active)
    @inbounds for I in CartesianIndices(occupied)
        iszero(occupied[I]) || _activate_block_neighborhood!(active, I, CI)
    end
    active
end

@inline function _inbounds_block(I::CartesianIndex{dim}, dims::Dims{dim}) where {dim}
    all(ntuple(d -> 1 ≤ I[d] ≤ dims[d], Val(dim)))
end

# GPU particle-driven updates expand occupied blocks here instead of relying on
# CPU ThreadPartition scheduling. Multiple threads may write the same `true`;
# only the final boolean state matters.
@kernel function gpukernel_expand_occupied_blocks!(active_blocks, @Const(occupied_blocks))
    b = @index(Global)
    @inbounds if !iszero(occupied_blocks[b])
        dims = size(occupied_blocks)
        blk = CartesianIndices(dims)[b]
        for offset in CartesianIndices(nfill(-1:1, Val(length(dims))))
            neighbor = CartesianIndex(ntuple(d -> blk[d] + offset[d], Val(length(dims))))
            if _inbounds_block(neighbor, dims)
                active_blocks[sub2ind(dims, neighbor)] = true
            end
        end
    end
end

function _activate_neighbor_blocks!(active, occupied, backend::GPU)
    fillzero!(active)
    expand_kernel = gpukernel_expand_occupied_blocks!(backend)
    expand_kernel(active, occupied; ndrange=length(occupied))
    active
end

update_sparsity!(::SpIndices, ::ThreadPartition{<: GPUBlockStrategy}) =
    error("update_sparsity! from a partition is CPU-only; on GPU update the sparsity from particle positions")
function update_sparsity!(spinds::SpIndices{dim, <:Any, <:Array{Int, dim}}, partition::ThreadPartition{<: BlockStrategy}) where {dim}
    bs = strategy(partition)
    nblocks(spinds) == nblocks(bs) || throw(ArgumentError("blocks per dimension $(nblocks(spinds)) must match"))
    block_size_log2(spinds) == block_size_log2(bs) ||
        throw(ArgumentError("block_size_log2 $(block_size_log2(spinds)) must match partition block_size_log2 $(block_size_log2(bs))"))

    activity = fillzero!(active_blocks(spinds))
    CI = CartesianIndices(activity)
    @inbounds for I in CI
        if !isempty(particle_indices(bs, I))
            _activate_block_neighborhood!(activity, I, CI)
        end
    end

    # Reuse numbering when the active block set is unchanged, mirroring the
    # particle-positions path: most steps move no block in or out of the set.
    _block_activity_unchanged(blocknumbering(spinds), activity) && return nothing
    _apply_block_activity!(spinds, activity)
end

function _block_activity_unchanged(numbers, activity)
    @inbounds for i in eachindex(numbers, activity)
        iszero(numbers[i]) == iszero(activity[i]) || return false
    end
    true
end
