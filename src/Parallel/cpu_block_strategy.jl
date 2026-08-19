# -----------------------------------------------------------------------------
#  CPUBlockStrategy
# -----------------------------------------------------------------------------

struct CPUBlockStrategy{dim, Mesh <: CartesianMesh{dim}} <: PartitionStrategy
    mesh::Mesh
    particleindices::Vector{Int}
    starts::Array{Int, dim}
    stops::Array{Int, dim}
    nassigned::Base.RefValue{Int}
    activegroups::Vector{Vector{CartesianIndex{dim}}}
    blockcolors::Array{Int, dim}
    update_workspace::BlockUpdateWorkspace{dim}
    matrix_buffer_pool::BlockMatrixBufferPool
    region_scratch::RegionScratch{Vector{CartesianIndex{dim}}}
end

function CPUBlockStrategy(mesh::CartesianMesh{dim}) where {dim}
    blkdims = nblocks(mesh)
    particleindices = Int[]
    starts = zeros(Int, blkdims)
    stops = zeros(Int, blkdims)
    activegroups = [CartesianIndex{dim}[] for _ in 1:(1 << dim)]
    blockcolors = zeros(Int, blkdims)
    for blk in CartesianIndices(blkdims)
        blockcolors[blk] = block_color(blk)
    end
    CPUBlockStrategy{dim, typeof(mesh)}(
        mesh,
        particleindices,
        starts,
        stops,
        Ref(0),
        activegroups,
        blockcolors,
        BlockUpdateWorkspace(blkdims),
        BlockMatrixBufferPool(),
        RegionScratch{Vector{CartesianIndex{dim}}}(),
    )
end

nblocks(bs::CPUBlockStrategy) = size(bs.stops)
block_size_log2(bs::CPUBlockStrategy) = block_size_log2(bs.mesh)
blockwidth(bs::CPUBlockStrategy) = blockwidth(bs.mesh)
nassigned(bs::CPUBlockStrategy) = bs.nassigned[]

@inline function _particle_indices(particleindices, starts, stops, blk::Integer)
    @_propagate_inbounds_meta
    start = starts[blk]
    stop = stops[blk]
    (iszero(start) || stop < start) && return view(particleindices, 1:0)
    view(particleindices, start:stop)
end
@inline function particle_indices(bs::CPUBlockStrategy, blk::Integer)
    @boundscheck checkbounds(LinearIndices(nblocks(bs)), blk)
    @inbounds _particle_indices(bs.particleindices, bs.starts, bs.stops, blk)
end
@inline function particle_indices(bs::CPUBlockStrategy, blk::CartesianIndex)
    @boundscheck checkbounds(CartesianIndices(nblocks(bs)), blk)
    @inbounds particle_indices(bs, LinearIndices(nblocks(bs))[blk])
end
function update!(bs::CPUBlockStrategy, xₚ::AbstractVector{<: Vec})
    nₚ = length(xₚ)
    chunksize = prepare_partition_update!(bs, nₚ)
    blocklin = LinearIndices(nblocks(bs))

    count_particles_by_block!(bs, xₚ, chunksize, blocklin)

    accumulate_chunk_counts!(bs)
    update_threadsafe_groups!(bs)
    assign_block_ranges!(bs)

    scatter_particle_indices!(bs, nₚ, chunksize)
    bs
end

function prepare_partition_update!(bs::CPUBlockStrategy, nₚ::Integer)
    ws = bs.update_workspace
    resize!(bs.particleindices, nₚ)
    resize!(ws.packed_particle_blocks, nₚ)
    check_packed_block_number_limits!(bs, nₚ)
    fillzero!(bs.starts)
    fillzero!(bs.stops)

    nchunks = length(ws.chunk_counts)
    max(1, cld(nₚ, nchunks))
end

function count_particles_by_block!(bs::CPUBlockStrategy, xₚ, chunksize, blocklin)
    ws = bs.update_workspace
    nₚ = length(xₚ)
    xmin = get_xmin(bs.mesh)
    h_inv = spacing_inv(bs.mesh)
    dims = size(bs.mesh)
    block_size = Val(block_size_log2(bs))

    @threaded for chunk_id in eachindex(ws.chunk_counts)
        # Zeroed by the worker that owns the histogram: doing it serially before
        # this loop is O(nthreads x nblocks) work that grows with `-t`.
        counts = fillzero!(ws.chunk_counts[chunk_id])

        @inbounds for p in chunk_range(chunk_id, chunksize, nₚ)
            blk = sub2ind(blocklin, _findblock(xₚ[p], xmin, h_inv, dims, block_size))
            if iszero(blk)
                ws.packed_particle_blocks[p] = 0
            else
                count = counts[blk] + 1
                counts[blk] = count
                ws.packed_particle_blocks[p] = pack_block_number(blk, count)
            end
        end
    end

    bs
end

# Accumulate chunk by chunk so each pass adds two contiguous arrays; a sweep
# carrying one block's running total through the chunks is strided instead.
function accumulate_chunk_counts!(bs::CPUBlockStrategy)
    ws = bs.update_workspace
    nchunks = length(ws.chunk_counts)

    @inbounds for chunk_id in 2:nchunks
        counts = ws.chunk_counts[chunk_id]
        prev_counts = ws.chunk_counts[chunk_id - 1]
        broadcast!(+, counts, counts, prev_counts)
    end

    bs
end

function assign_block_ranges!(bs::CPUBlockStrategy)
    ws = bs.update_workspace
    blocklin = LinearIndices(nblocks(bs))
    last_counts = ws.chunk_counts[end]

    @inbounds begin
        assigned = 0
        for group in bs.activegroups
            for region in group
                blk = blocklin[region]
                bs.starts[blk] = assigned + 1
                count = last_counts[blk]
                assigned += count
                bs.stops[blk] = assigned
            end
        end
        bs.nassigned[] = assigned
    end

    bs
end

const PACKED_BLOCK_NUMBER_BITS = 32
const PACKED_BLOCK_NUMBER_MASK = (UInt64(1) << PACKED_BLOCK_NUMBER_BITS) - UInt64(1)

function check_packed_block_number_limits!(bs::CPUBlockStrategy, nₚ::Integer)
    block_count = foldl((count, n) -> count * UInt64(n), nblocks(bs); init = UInt64(1))
    block_count <= PACKED_BLOCK_NUMBER_MASK ||
        throw(ArgumentError("Partition block count exceeds packed block id capacity."))
    UInt64(nₚ) <= PACKED_BLOCK_NUMBER_MASK ||
        throw(ArgumentError("Partition particle count exceeds packed per-block number capacity."))
    nothing
end

# The linear block id rides in the upper 32 bits, the 1-based number within the
# particle's chunk/block in the lower 32. The chunk id is not stored: count and
# scatter walk the same index ranges.
@inline pack_block_number(block::Integer, number::Integer) =
    (UInt64(block) << PACKED_BLOCK_NUMBER_BITS) | UInt64(number)
@inline packed_block(packed::UInt64) = Int(packed >> PACKED_BLOCK_NUMBER_BITS)
@inline packed_number(packed::UInt64) = Int(packed & PACKED_BLOCK_NUMBER_MASK)

function scatter_particle_indices!(bs::CPUBlockStrategy, nₚ::Integer, chunksize)
    ws = bs.update_workspace

    @threaded for chunk_id in eachindex(ws.chunk_counts)
        @inbounds for p in chunk_range(chunk_id, chunksize, nₚ)
            packed = ws.packed_particle_blocks[p]
            if !iszero(packed)
                blk = packed_block(packed)
                chunk_offset = isone(chunk_id) ? 0 : ws.chunk_counts[chunk_id - 1][blk]
                number = chunk_offset + packed_number(packed)
                bs.particleindices[bs.starts[blk] + number - 1] = p
            end
        end
    end

    bs
end

@inline sub2ind(dims::Dims, I)::Int = @inbounds LinearIndices(dims)[I]
@inline sub2ind(::Dims, ::Nothing)::Int = 0
@inline sub2ind(lin::LinearIndices, I::CartesianIndex)::Int = @inbounds lin[I]
@inline sub2ind(::LinearIndices, ::Nothing)::Int = 0

@inline function block_color(I::CartesianIndex{dim}) where {dim}
    color = 1
    @inbounds for d in 1:dim
        color += ((I[d] - 1) & 1) << (d - 1)
    end
    color
end

function update_threadsafe_groups!(bs::CPUBlockStrategy)
    for active in bs.activegroups
        empty!(active)
    end
    ws = bs.update_workspace
    counts = ws.chunk_counts[end]
    cart = CartesianIndices(nblocks(bs))
    @inbounds for blk in eachindex(counts)
        if !iszero(counts[blk])
            region = cart[blk]
            push!(bs.activegroups[bs.blockcolors[blk]], region)
        end
    end
    bs.activegroups
end
threadsafe_groups(bs::CPUBlockStrategy) = bs.activegroups

"""
    Tesserae.block_ordered_particle_contiguity(partition)

Score in `[0, 1]` for how block-ordered the particles are in memory: `1` just
after `reorder_particles!`, decreasing as particles move across blocks. The
score means the same thing on every backend and compares against the same
threshold; only the estimator behind it differs.

On a CPU partition the score is the fraction of neighboring entries in the
current block-ordered particle index array that are also consecutive in
memory. For example, a block-ordered list `[1, 2, 3, 8]` has two consecutive
pairs out of three. On a GPU partition, whose scatter randomizes the order
within each block, it is the fraction of neighboring particles in memory that
share a block, rescaled so that fully block-grouped storage scores `1`.
"""
function block_ordered_particle_contiguity(bs::CPUBlockStrategy)
    n_assigned = nassigned(bs)
    n_assigned ≤ 1 && return 1.0

    consecutive = 0
    @inbounds for i in 2:n_assigned
        consecutive += bs.particleindices[i] == bs.particleindices[i-1] + 1
    end
    consecutive / (n_assigned - 1)
end

"""
    reorder_particles!(particles, partition; threshold=1)

Reorder particles by the current block partition, and return whether it did.

Particles are reordered when [`Tesserae.block_ordered_particle_contiguity`](@ref)
is below `threshold`, which by default is every call. For `0 ≤ threshold ≤ 1`,
larger values reorder more often; `threshold=0` never reorders.

In a step loop, call this every step but pass a `threshold` below `1`, such as
`0.85`, and let it decide which steps to act on. Reordering moves about as many
bytes as the transfer it speeds up, so reordering on every step usually costs
more than it saves.

On a partition moved with `gpu`, the reorder runs on the device through the
partition's block-sorted permutation. Unlike the CPU path, particles outside
the mesh are an error there rather than being kept at the end of the array.

!!! warning
    This permutes `particles` and nothing else, so anything already computed per
    particle -- basis weights above all -- is stale afterwards. Call it before
    `update!(weights, particles, mesh)`, not between that and the transfer.
"""
function reorder_particles!(particles::StructVector, bs::CPUBlockStrategy; threshold=1)
    0 ≤ threshold ≤ 1 || throw(ArgumentError("threshold must be in [0, 1]."))
    iszero(threshold) && return false
    if threshold == 1 || block_ordered_particle_contiguity(bs) < threshold
        _reorder_partition_particles!(particles, bs)
        return true
    end
    return false
end

function _reorder_partition_particles!(particles::StructVector, bs::CPUBlockStrategy)
    n_assigned = nassigned(bs)
    _reorder_particles!(particles, bs.particleindices, n_assigned, bs.update_workspace.particle_reorder_buffers)
    copyto!(bs.particleindices, 1:n_assigned)
    particles
end

# `buffer_for_component!` returns one buffer per element type, so components of
# the same type share it; each must be copied back before the next is gathered.
function _permute_particles!(particles::StructVector, perm, buffers::ParticleReorderBuffers)
    for component in StructArrays.components(particles)
        buffer = buffer_for_component!(buffers, component)
        _permute_component!(component, perm, buffer)
    end
    particles
end

# `_reorder_particles!` passes a full-length valid permutation and the buffer
# matches the component, so the gather can skip bounds checks. Sequential on
# purpose: threading costs two fork-joins per component and measured slower at
# every size tried. The copy back stays a separate pass, the gather still reading
# elements that copying would overwrite.
function _permute_component!(component, perm, buffer)
    n = length(component)
    @inbounds for k in 1:n
        buffer[k] = component[perm[k]]
    end
    copyto!(component, buffer)
    component
end

function _reorder_particles!(particles::StructVector, particleindices::AbstractVector{Int}, nₚ_assigned::Integer, buffers::ParticleReorderBuffers=ParticleReorderBuffers())
    nₚ = length(particles)

    (firstindex(particles) == 1 && lastindex(particles) == nₚ) || throw(ArgumentError("reorder_particles!: particles must be 1-based indexed (`Vector`-like)."))
    nₚ_assigned > nₚ && error("reorder_particles!: The block assignment contains more particle IDs than exist (assigned=$nₚ_assigned, total=$nₚ).")

    # With every particle assigned, particleindices is already the permutation.
    if nₚ_assigned == nₚ
        particle_order = view(particleindices, 1:nₚ)
        _permute_particles!(particles, particle_order, buffers)
        return particle_order
    end

    perm = Vector{Int}(undef, nₚ)
    # Only the first nₚ_assigned entries are valid; the rest may contain stale
    # ids from a previous partition update.
    copyto!(perm, 1, particleindices, 1, nₚ_assigned)

    # Fallback: keep particles outside the mesh after the assigned particles,
    # preserving their original relative order.
    seen = falses(nₚ)
    for i in 1:nₚ_assigned
        p = perm[i]
        1 ≤ p ≤ nₚ || error("reorder_particles!: particle ID $p is out of range (valid: 1:$nₚ).")
        @inbounds begin
            seen[p] && error("reorder_particles!: particle $p is duplicated in the block assignment.")
            seen[p] = true
        end
    end

    @warn "reorder_particles!: Some particles are outside of the grid and were not assigned to any block. They will be kept at the end of the array." maxlog=1
    k = nₚ_assigned
    @inbounds for p in 1:nₚ
        if !seen[p]
            k += 1
            perm[k] = p
        end
    end
    @assert k == nₚ

    _permute_particles!(particles, perm, buffers)

    perm
end
