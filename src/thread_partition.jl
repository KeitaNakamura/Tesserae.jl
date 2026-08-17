# -----------------------------------------------------------------------------
#  Thread partitioning
# -----------------------------------------------------------------------------

# ---- BlockStrategy ----

abstract type PartitionStrategy end

struct ParticleReorderBuffers
    by_component_type::Dict{DataType, Any}
end
ParticleReorderBuffers() = ParticleReorderBuffers(Dict{DataType, Any}())

function buffer_for_component!(buffers::ParticleReorderBuffers, component::T) where {T}
    buffer = get(buffers.by_component_type, T, nothing)
    if !(buffer isa T) || length(buffer) != length(component)
        buffer = similar(component, length(component))
        buffers.by_component_type[T] = buffer
    end
    buffer::T
end

struct BlockUpdateWorkspace{dim}
    chunk_counts::Vector{Array{Int, dim}}   # per-chunk block histogram
    packed_particle_blocks::Vector{UInt64}  # block id and number within chunk/block
    particle_reorder_buffers::ParticleReorderBuffers
end

function BlockUpdateWorkspace(blkdims::Dims{dim}) where {dim}
    BlockUpdateWorkspace{dim}(
        [zeros(Int, blkdims) for _ in 1:Threads.nthreads()],
        UInt64[],
        ParticleReorderBuffers(),
    )
end

struct BlockMatrixBufferPool
    lock::ReentrantLock
    buffers::Dict{Any, Vector{Any}}
end
BlockMatrixBufferPool() = BlockMatrixBufferPool(ReentrantLock(), Dict{Any, Vector{Any}}())

struct BlockStrategy{dim, Mesh <: CartesianMesh{dim}} <: PartitionStrategy
    mesh::Mesh
    particleindices::Vector{Int}
    starts::Array{Int, dim}
    stops::Array{Int, dim}
    nassigned::Base.RefValue{Int}
    activegroups::Vector{Vector{CartesianIndex{dim}}}
    blockcolors::Array{Int, dim}
    update_workspace::BlockUpdateWorkspace{dim}
    matrix_buffer_pool::BlockMatrixBufferPool
end

function BlockStrategy(mesh::CartesianMesh{dim}) where {dim}
    blkdims = nblocks(mesh)
    particleindices = Int[]
    starts = zeros(Int, blkdims)
    stops = zeros(Int, blkdims)
    activegroups = [CartesianIndex{dim}[] for _ in 1:(1 << dim)]
    blockcolors = zeros(Int, blkdims)
    for blk in CartesianIndices(blkdims)
        blockcolors[blk] = block_color(blk)
    end
    BlockStrategy{dim, typeof(mesh)}(
        mesh,
        particleindices,
        starts,
        stops,
        Ref(0),
        activegroups,
        blockcolors,
        BlockUpdateWorkspace(blkdims),
        BlockMatrixBufferPool(),
    )
end

nblocks(bs::BlockStrategy) = size(bs.stops)
block_size_log2(bs::BlockStrategy) = block_size_log2(bs.mesh)
blockwidth(bs::BlockStrategy) = blockwidth(bs.mesh)
nassigned(bs::BlockStrategy) = bs.nassigned[]

@inline function _particle_indices(particleindices, starts, stops, blk::Integer)
    @_propagate_inbounds_meta
    start = starts[blk]
    stop = stops[blk]
    (iszero(start) || stop < start) && return view(particleindices, 1:0)
    view(particleindices, start:stop)
end
@inline function particle_indices(bs::BlockStrategy, blk::Integer)
    @boundscheck checkbounds(LinearIndices(nblocks(bs)), blk)
    @inbounds _particle_indices(bs.particleindices, bs.starts, bs.stops, blk)
end
@inline function particle_indices(bs::BlockStrategy, blk::CartesianIndex)
    @boundscheck checkbounds(CartesianIndices(nblocks(bs)), blk)
    @inbounds particle_indices(bs, LinearIndices(nblocks(bs))[blk])
end
function update!(bs::BlockStrategy, xₚ::AbstractVector{<: Vec})
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

function prepare_partition_update!(bs::BlockStrategy, nₚ::Integer)
    ws = bs.update_workspace
    resize!(bs.particleindices, nₚ)
    resize!(ws.packed_particle_blocks, nₚ)
    check_packed_block_number_limits!(bs, nₚ)
    foreach(fillzero!, ws.chunk_counts)
    fillzero!(bs.starts)
    fillzero!(bs.stops)

    nchunks = length(ws.chunk_counts)
    max(1, cld(nₚ, nchunks))
end

function count_particles_by_block!(bs::BlockStrategy, xₚ, chunksize, blocklin)
    ws = bs.update_workspace
    nₚ = length(xₚ)
    xmin = get_xmin(bs.mesh)
    h_inv = spacing_inv(bs.mesh)
    dims = size(bs.mesh)
    block_size = Val(block_size_log2(bs))

    @threaded for chunk_id in eachindex(ws.chunk_counts)
        counts = ws.chunk_counts[chunk_id]

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
function accumulate_chunk_counts!(bs::BlockStrategy)
    ws = bs.update_workspace
    nchunks = length(ws.chunk_counts)

    @inbounds for chunk_id in 2:nchunks
        counts = ws.chunk_counts[chunk_id]
        prev_counts = ws.chunk_counts[chunk_id - 1]
        broadcast!(+, counts, counts, prev_counts)
    end

    bs
end

function assign_block_ranges!(bs::BlockStrategy)
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

function check_packed_block_number_limits!(bs::BlockStrategy, nₚ::Integer)
    block_count = foldl((count, n) -> count * UInt64(n), nblocks(bs); init = UInt64(1))
    block_count <= PACKED_BLOCK_NUMBER_MASK ||
        throw(ArgumentError("ThreadPartition block count exceeds packed block id capacity."))
    UInt64(nₚ) <= PACKED_BLOCK_NUMBER_MASK ||
        throw(ArgumentError("ThreadPartition particle count exceeds packed per-block number capacity."))
    nothing
end

# The linear block id rides in the upper 32 bits, the 1-based number within the
# particle's chunk/block in the lower 32. The chunk id is not stored: count and
# scatter walk the same index ranges.
@inline pack_block_number(block::Integer, number::Integer) =
    (UInt64(block) << PACKED_BLOCK_NUMBER_BITS) | UInt64(number)
@inline packed_block(packed::UInt64) = Int(packed >> PACKED_BLOCK_NUMBER_BITS)
@inline packed_number(packed::UInt64) = Int(packed & PACKED_BLOCK_NUMBER_MASK)

function scatter_particle_indices!(bs::BlockStrategy, nₚ::Integer, chunksize)
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

function update_threadsafe_groups!(bs::BlockStrategy)
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
threadsafe_groups(bs::BlockStrategy) = bs.activegroups

"""
    Tesserae.block_ordered_particle_contiguity(partition)

Return how contiguous the block-ordered particle list is in memory order.
The score is `1` just after `reorder_particles!` and decreases as particles
move across blocks.

The score is the fraction of neighboring entries in the current block-ordered
particle index array that are also consecutive in memory. For example, a
block-ordered list `[1, 2, 3, 8]` has two consecutive pairs out of three.
"""
function block_ordered_particle_contiguity(bs::BlockStrategy)
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

!!! warning
    This permutes `particles` and nothing else, so anything already computed per
    particle -- basis weights above all -- is stale afterwards. Call it before
    `update!(weights, particles, mesh)`, not between that and the transfer.
"""
function reorder_particles!(particles::StructVector, bs::BlockStrategy; threshold=1)
    0 ≤ threshold ≤ 1 || throw(ArgumentError("threshold must be in [0, 1]."))
    iszero(threshold) && return false
    if threshold == 1 || block_ordered_particle_contiguity(bs) < threshold
        _reorder_partition_particles!(particles, bs)
        return true
    end
    return false
end

function _reorder_partition_particles!(particles::StructVector, bs::BlockStrategy)
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

# ---- block operations ----

blockwidth(::Val{L}) where {L} = 1 << L
blockwidth(mesh::CartesianMesh) = blockwidth(Val(block_size_log2(mesh)))

nblocks(gridsize::Tuple{Vararg{Int}}; block_size_log2::Val{L}) where {L} =
    (_check_block_size_log2(block_size_log2); map(n -> ((n - 1) >> L) + 1, gridsize))
nblocks(mesh::CartesianMesh) = nblocks(size(mesh); block_size_log2=Val(block_size_log2(mesh)))

"""
    Tesserae.findblock(x::Vec, mesh::CartesianMesh)

Return block index where `x` locates.
The unit block size is `2^block_size_log2(mesh)` cells.

# Examples
```jldoctest
julia> mesh = CartesianMesh(1, (0,10), (0,10))
11×11 CartesianMesh{2, Float64, Vector{Float64}, 2}:
 [0.0, 0.0]   [0.0, 1.0]   [0.0, 2.0]   …  [0.0, 9.0]   [0.0, 10.0]
 [1.0, 0.0]   [1.0, 1.0]   [1.0, 2.0]      [1.0, 9.0]   [1.0, 10.0]
 [2.0, 0.0]   [2.0, 1.0]   [2.0, 2.0]      [2.0, 9.0]   [2.0, 10.0]
 [3.0, 0.0]   [3.0, 1.0]   [3.0, 2.0]      [3.0, 9.0]   [3.0, 10.0]
 [4.0, 0.0]   [4.0, 1.0]   [4.0, 2.0]      [4.0, 9.0]   [4.0, 10.0]
 [5.0, 0.0]   [5.0, 1.0]   [5.0, 2.0]   …  [5.0, 9.0]   [5.0, 10.0]
 [6.0, 0.0]   [6.0, 1.0]   [6.0, 2.0]      [6.0, 9.0]   [6.0, 10.0]
 [7.0, 0.0]   [7.0, 1.0]   [7.0, 2.0]      [7.0, 9.0]   [7.0, 10.0]
 [8.0, 0.0]   [8.0, 1.0]   [8.0, 2.0]      [8.0, 9.0]   [8.0, 10.0]
 [9.0, 0.0]   [9.0, 1.0]   [9.0, 2.0]      [9.0, 9.0]   [9.0, 10.0]
 [10.0, 0.0]  [10.0, 1.0]  [10.0, 2.0]  …  [10.0, 9.0]  [10.0, 10.0]

julia> Tesserae.findblock(Vec(8.5, 1.5), mesh)
CartesianIndex(3, 1)
```
"""
@inline function findblock(x::Vec{dim}, mesh::CartesianMesh{dim, T, V, L}) where {dim, T, V, L}
    _findblock(x, get_xmin(mesh), spacing_inv(mesh), size(mesh), Val(L))
end

# Same boundary rule as `findcell`, but returning the block index directly: the
# 0-based cell index shifted by `block_size_log2`, plus one.
@generated function _findblock(x::Vec{dim}, xmin::Vec{dim}, h_inv, dims::Dims{dim}, ::Val{L}) where {dim, L}
    quote
        @_inline_meta
        @nexprs $dim d -> cell0_d = unsafe_trunc(Int, floor((x[d] - xmin[d]) * h_inv))
        inside = @nall $dim d -> 0 ≤ cell0_d ≤ dims[d] - 2
        inside || return nothing
        CartesianIndex(@ntuple $dim d -> (cell0_d >> $L) + 1)
    end
end

# GPU sibling of `BlockStrategy`, maintained on the device by a counting sort so
# no host synchronization happens beyond one readback of the active-block count.
# Particle order within a block is whatever the atomic scatter produced, which
# permutes a floating-point sum that particle motion reorders anyway.
# ---- GPUBlockStrategy ----

struct GPUBlockStrategy{dim, Mesh <: CartesianMesh{dim}, Vi <: AbstractVector{Int32}, Vl <: AbstractVector{Int64}} <: PartitionStrategy
    mesh::Mesh
    particleindices::Vi  # block-contiguous particle ids
    blockids::Vi         # per-particle linear block id, 0 while outside the mesh
    counts::Vi           # per-block particle count
    offsets::Vi          # exclusive prefix of counts, length nblocks+1
    cursors::Vi          # per-block scatter cursors
    partials::Vl         # per-workgroup partial sums for the fused scan
    blocklist::Vi        # nonempty block ids; the first `nactive` entries are valid
    nactive_buf::Vi      # device-side active-block count
    nactive::Base.RefValue{Int}
end

# Count prefix and nonempty-block prefix ride in one Int64, counts in the low
# half and flags in the high; both totals stay below 2^31, so neither carries.
const PARTITION_SCAN_GROUP = 256
@inline pack_partition_sums(count::Int32) = Int64(count) | (Int64(!iszero(count)) << 32)
@inline packed_count(x::Int64) = x & Int64(0xffffffff)
@inline packed_flags(x::Int64) = x >> 32

function GPUBlockStrategy(mesh::CartesianMesh)
    backend = get_backend(mesh)
    nb = prod(nblocks(mesh))
    alloc(::Type{T}, n) where {T} = fillzero!(KernelAbstractions.allocate(backend, T, n))
    GPUBlockStrategy(
        mesh, alloc(Int32, 0), alloc(Int32, 0), alloc(Int32, nb), alloc(Int32, nb + 1), alloc(Int32, nb),
        alloc(Int64, cld(nb, PARTITION_SCAN_GROUP)), alloc(Int32, nb), alloc(Int32, 1), Ref(0),
    )
end

nblocks(bs::GPUBlockStrategy) = nblocks(bs.mesh)
block_size_log2(bs::GPUBlockStrategy) = block_size_log2(bs.mesh)
blockwidth(bs::GPUBlockStrategy) = blockwidth(bs.mesh)
nactive(bs::GPUBlockStrategy) = bs.nactive[]

@kernel function gpukernel_partition_count!(blockids, counts, @Const(xₚ), @Const(mesh))
    p = @index(Global)
    b = Int32(sub2ind(nblocks(mesh), findblock(xₚ[p], mesh)))
    @inbounds blockids[p] = b
    if !iszero(b)
        @inbounds Atomix.@atomic counts[b] += Int32(1)
    end
end

# Fused two-level scan: both prefixes the update needs ride in one packed Int64
# and the final rescan writes the per-block outputs directly. The generic
# `cumsum!` this replaces cost more in kernel launches than the particle work.
@kernel function gpukernel_partition_reduce!(partials, @Const(counts))
    g = @index(Group)
    l = @index(Local)
    b = @index(Global)
    sm = @localmem Int64 (PARTITION_SCAN_GROUP,)
    @inbounds sm[l] = b <= length(counts) ? pack_partition_sums(counts[b]) : Int64(0)
    @synchronize
    stride = PARTITION_SCAN_GROUP >> 1
    while stride > 0
        @inbounds l <= stride && (sm[l] += sm[l+stride])
        @synchronize
        stride >>= 1
    end
    @inbounds isone(l) && (partials[g] = sm[1])
end

# Single-workgroup exclusive scan of the per-group partials, in place.
@kernel function gpukernel_partition_spine!(partials)
    l = @index(Local)
    n = length(partials)
    sm = @localmem Int64 (PARTITION_SCAN_GROUP,)
    base = Int64(0)
    for chunk in 1:cld(n, PARTITION_SCAN_GROUP)
        b = (chunk - 1) * PARTITION_SCAN_GROUP + l
        v = Int64(0)
        @inbounds b <= n && (v = partials[b])
        @inbounds sm[l] = v
        @synchronize
        step = 1
        while step < PARTITION_SCAN_GROUP
            t = l > step ? (@inbounds sm[l-step]) : Int64(0)
            @synchronize
            @inbounds sm[l] += t
            @synchronize
            step <<= 1
        end
        @inbounds b <= n && (partials[b] = base + sm[l] - v)
        @inbounds base += sm[PARTITION_SCAN_GROUP]
        @synchronize
    end
end

# Every per-block output is written in this one rescan pass.
@kernel function gpukernel_partition_finalize!(offsets, cursors, blocklist, nactive_buf, @Const(counts), @Const(partials))
    g = @index(Group)
    l = @index(Local)
    b = @index(Global)
    nb = length(counts)
    sm = @localmem Int64 (PARTITION_SCAN_GROUP,)
    c = Int32(0)
    @inbounds b <= nb && (c = counts[b])
    @inbounds sm[l] = pack_partition_sums(c)
    @synchronize
    step = 1
    while step < PARTITION_SCAN_GROUP
        t = l > step ? (@inbounds sm[l-step]) : Int64(0)
        @synchronize
        @inbounds sm[l] += t
        @synchronize
        step <<= 1
    end
    @inbounds if b <= nb
        incl = partials[g] + sm[l]
        start = Int32(packed_count(incl) - Int64(c))
        offsets[b] = start
        cursors[b] = start
        iszero(c) || (blocklist[packed_flags(incl)] = Int32(b))
        if b == nb
            offsets[nb+1] = Int32(packed_count(incl))
            nactive_buf[1] = Int32(packed_flags(incl))
        end
    end
end

@kernel function gpukernel_partition_scatter!(particleindices, cursors, @Const(blockids))
    p = @index(Global)
    @inbounds b = blockids[p]
    if !iszero(b)
        pos = @inbounds Atomix.@atomic cursors[b] += Int32(1)
        @inbounds particleindices[pos] = Int32(p)
    end
end

function update!(bs::GPUBlockStrategy, xₚ::AbstractVector{<: Vec})
    backend = get_backend(bs.mesh)
    get_backend(xₚ) == backend || throw(ArgumentError("particle positions must live on the partition's backend"))
    nₚ = length(xₚ)
    # Int32 particle ids, and both packed scan halves must stay below 2^31.
    nₚ <= typemax(Int32) || throw(ArgumentError("ThreadPartition: particle count exceeds the GPU partition's Int32 capacity"))
    length(bs.counts) <= typemax(Int32) || throw(ArgumentError("ThreadPartition: block count exceeds the GPU partition's Int32 capacity"))
    ngroups = length(bs.partials)
    length(bs.blockids) == nₚ || resize_fillzero!(bs.blockids, nₚ)
    length(bs.particleindices) == nₚ || resize_fillzero!(bs.particleindices, nₚ)
    fillzero!(bs.counts)

    iszero(nₚ) || gpukernel_partition_count!(backend)(bs.blockids, bs.counts, xₚ, bs.mesh; ndrange=nₚ)
    scan_ndrange = ngroups * PARTITION_SCAN_GROUP
    gpukernel_partition_reduce!(backend, PARTITION_SCAN_GROUP)(bs.partials, bs.counts; ndrange=scan_ndrange)
    gpukernel_partition_spine!(backend, PARTITION_SCAN_GROUP)(bs.partials; ndrange=PARTITION_SCAN_GROUP)
    gpukernel_partition_finalize!(backend, PARTITION_SCAN_GROUP)(bs.offsets, bs.cursors, bs.blocklist, bs.nactive_buf, bs.counts, bs.partials; ndrange=scan_ndrange)
    iszero(nₚ) || gpukernel_partition_scatter!(backend)(bs.particleindices, bs.cursors, bs.blockids; ndrange=nₚ)

    KernelAbstractions.synchronize(backend)
    bs.nactive[] = Int(only(Array(bs.nactive_buf)))
    bs
end

# ---- CellStrategy ----

struct CellStrategy <: PartitionStrategy
    threadsafe_groups::Vector{Vector{Int}}
end

threadsafe_groups(cs::CellStrategy) = cs.threadsafe_groups

function CellStrategy(mesh::AbstractCellMesh)
    g = _cell_conflict_graph(mesh)

    coloring = Graphs.degree_greedy_color(g)

    groups = [Int[] for _ in 1:coloring.num_colors]
    @inbounds for (cellid, cell) in enumerate(cells(mesh))
        push!(groups[coloring.colors[cellid]], cellid)
    end

    CellStrategy(groups)
end

function _cell_conflict_graph(mesh::AbstractCellMesh)
    nc = ncells(mesh)
    nn = length(mesh)
    graph = SimpleGraph(nc)

    node2cells = [Int[] for _ in 1:nn]
    @inbounds for (cellid, cell) in enumerate(cells(mesh))
        for i in supportnodes(mesh, cell)
            push!(node2cells[i], cellid)
        end
    end

    for cells in node2cells
        m = length(cells)
        @inbounds for i in 1:m-1
            cell = cells[i]
            for j in i+1:m
                add_edge!(graph, cell, cells[j])
            end
        end
    end

    graph
end

# ---- ThreadPartition ----

"""
    ThreadPartition(::CartesianMesh)
    ThreadPartition(::FEMesh)
    ThreadPartition(::IGAMesh)

`ThreadPartition` stores partitioning information used by the [`@P2G`](@ref), [`@G2P2G`](@ref) and [`@P2G_Matrix`](@ref) macros
to avoid write conflicts during threaded particle-to-grid transfers.

On GPU the same type schedules workgroups instead of threads: `gpu(partition)`
rebuilds it for the device, and passing it to [`@P2G`](@ref) selects a
block-scheduled kernel that accumulates each grid block in shared memory. There
`@threaded` plays no part, and [`@G2P2G`](@ref) and [`@P2G_Matrix`](@ref) do not
take a device partition yet.

!!! note
    The [`@threaded`](@ref) macro must be placed before [`@P2G`](@ref), [`@G2P2G`](@ref) and [`@P2G_Matrix`](@ref) to enable parallel transfer on CPU.

# Examples
```julia
# Construct ThreadPartition
partition = ThreadPartition(mesh)

# Update partition using current particle positions
update!(partition, particles.x) # Required only for `CartesianMesh`.

# P2G transfer
@threaded @P2G grid=>i particles=>p weights=>ip partition begin
    m[i]  = @∑ w[ip] * m[p]
    mv[i] = @∑ w[ip] * m[p] * v[p]
end
```
"""
struct ThreadPartition{Strategy <: PartitionStrategy}
    strategy::Strategy
end

strategy(partition::ThreadPartition) = partition.strategy
threadsafe_groups(partition::ThreadPartition) = threadsafe_groups(strategy(partition))

particle_indices(partition::ThreadPartition, particles, region) =
    particle_indices(strategy(partition), region)
particle_indices(partition::ThreadPartition{<: CellStrategy}, particles, cell) =
    (CartesianIndex(p, cell) for p in 1:size(particles, 1))

ThreadPartition(mesh::CartesianMesh) = ThreadPartition(BlockStrategy(mesh))
ThreadPartition(mesh::AbstractCellMesh) = ThreadPartition(CellStrategy(mesh))
update!(partition::ThreadPartition, args...) = update!(strategy(partition), args...)

reorder_particles!(particles::StructVector, partition::ThreadPartition{<: BlockStrategy}; kwargs...) =
    reorder_particles!(particles, strategy(partition); kwargs...)
block_ordered_particle_contiguity(partition::ThreadPartition{<: BlockStrategy}) =
    block_ordered_particle_contiguity(strategy(partition))
reorder_particles!(particles, ::ThreadPartition{<: GPUBlockStrategy}; kwargs...) =
    error("reorder_particles! does not support GPU partitions yet")
block_ordered_particle_contiguity(::ThreadPartition{<: GPUBlockStrategy}) =
    error("block_ordered_particle_contiguity does not support GPU partitions yet")
