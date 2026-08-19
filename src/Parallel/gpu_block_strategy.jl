# -----------------------------------------------------------------------------
#  GPUBlockStrategy
# -----------------------------------------------------------------------------

# GPU sibling of `CPUBlockStrategy`, maintained on the device by a counting sort so
# no host synchronization happens beyond one readback of the block and particle totals.
# Particle order within a block is whatever the atomic scatter produced, which
# permutes a floating-point sum that particle motion reorders anyway.

struct GPUBlockStrategy{dim, Mesh <: CartesianMesh{dim}, Vi <: AbstractVector{Int32}, Vl <: AbstractVector{Int64}} <: PartitionStrategy
    mesh::Mesh
    particleindices::Vi  # block-contiguous particle ids
    blockids::Vi         # per-particle linear block id, 0 while outside the mesh
    counts::Vi           # per-block particle count
    offsets::Vi          # exclusive prefix of counts, length nblocks+1
    cursors::Vi          # per-block scatter cursors
    partials::Vl         # per-workgroup partial sums for the fused scan
    blocklist::Vi        # nonempty block ids; the first `nactive` entries are valid
    totals_buf::Vi       # device-side [active-block count, assigned-particle count]
    nactive::Base.RefValue{Int}
    nassigned::Base.RefValue{Int}
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
        alloc(Int64, cld(nb, PARTITION_SCAN_GROUP)), alloc(Int32, nb), alloc(Int32, 2), Ref(0), Ref(0),
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
@kernel function gpukernel_partition_finalize!(offsets, cursors, blocklist, totals_buf, @Const(counts), @Const(partials))
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
            totals_buf[1] = Int32(packed_flags(incl))
            totals_buf[2] = Int32(packed_count(incl))
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
    nₚ <= typemax(Int32) || throw(ArgumentError("Partition: particle count exceeds the GPU partition's Int32 capacity"))
    length(bs.counts) <= typemax(Int32) || throw(ArgumentError("Partition: block count exceeds the GPU partition's Int32 capacity"))
    ngroups = length(bs.partials)
    length(bs.blockids) == nₚ || resize_fillzero!(bs.blockids, nₚ)
    length(bs.particleindices) == nₚ || resize_fillzero!(bs.particleindices, nₚ)
    fillzero!(bs.counts)

    iszero(nₚ) || gpukernel_partition_count!(backend)(bs.blockids, bs.counts, xₚ, bs.mesh; ndrange=nₚ)
    scan_ndrange = ngroups * PARTITION_SCAN_GROUP
    gpukernel_partition_reduce!(backend, PARTITION_SCAN_GROUP)(bs.partials, bs.counts; ndrange=scan_ndrange)
    gpukernel_partition_spine!(backend, PARTITION_SCAN_GROUP)(bs.partials; ndrange=PARTITION_SCAN_GROUP)
    gpukernel_partition_finalize!(backend, PARTITION_SCAN_GROUP)(bs.offsets, bs.cursors, bs.blocklist, bs.totals_buf, bs.counts, bs.partials; ndrange=scan_ndrange)
    iszero(nₚ) || gpukernel_partition_scatter!(backend)(bs.particleindices, bs.cursors, bs.blockids; ndrange=nₚ)

    KernelAbstractions.synchronize(backend)
    totals = Array(bs.totals_buf)
    bs.nactive[] = Int(totals[1])
    bs.nassigned[] = Int(totals[2])
    bs
end

# The scatter above fills slots 1:nassigned contiguously; the total rides home
# in the same readback that fetches `nactive`, so reading it here costs no
# device round-trip on the adaptive reorder path.
nassigned(bs::GPUBlockStrategy) = bs.nassigned[]

block_ordered_particle_contiguity(bs::GPUBlockStrategy) = same_block_neighbor_fraction(bs)

# The estimator behind the public score on GPU: the device scatter randomizes
# order within a block, so the CPU estimator (adjacency of the block-ordered
# indices) reads as disorder even right after a reorder. Count instead the
# neighboring particles that share a block, which that shuffle cannot
# disturb: 1 after a reorder, decreasing as particles change blocks.
function same_block_neighbor_fraction(bs::GPUBlockStrategy)
    nₚ = length(bs.blockids)
    nₚ ≤ 1 && return 1.0
    blockids = bs.blockids
    # Block slot ranges shift globally whenever any upstream block changes its
    # count, so judging particles against them would misreport still-grouped
    # particles as disordered; only the assigned total comes from `offsets`.
    same = mapreduce((a, b) -> Int(!iszero(a) & (a == b)), +,
                     view(blockids, 1:nₚ-1), view(blockids, 2:nₚ); init=0)
    maxsame = nassigned(bs) - nactive(bs)
    maxsame ≤ 0 && return 1.0
    same / maxsame
end

function reorder_particles!(particles::StructVector, bs::GPUBlockStrategy; threshold=1)
    0 ≤ threshold ≤ 1 || throw(ArgumentError("threshold must be in [0, 1]."))
    iszero(threshold) && return false
    if threshold == 1 || block_ordered_particle_contiguity(bs) < threshold
        _reorder_partition_particles!(particles, bs)
        return true
    end
    return false
end

function _reorder_partition_particles!(particles::StructVector, bs::GPUBlockStrategy)
    nₚ = length(particles)
    # Before the buffer-length guard: Metal keeps a stale nonzero buffer when
    # resized to empty, which would misreport a followed contract as a missing
    # `update!`.
    iszero(nₚ) && return particles
    length(bs.particleindices) == nₚ ||
        error("reorder_particles!: `update!(partition, particles.x)` must run with these particles before reordering")
    # Unlike the CPU path there is no append fallback: collecting the unassigned
    # ids needs a host-side pass over every particle, so stray particles error.
    nassigned(bs) == nₚ ||
        error("reorder_particles!: some particles are outside the mesh and were not assigned to any block; filter them out before reordering on GPU")
    perm = bs.particleindices
    for component in StructArrays.components(particles)
        permute_through_temporary!(component, perm)
    end
    # `blockids` follows the same permutation so the contiguity metric stays
    # truthful until the next `update!` recomputes it.
    permute_through_temporary!(bs.blockids, perm)
    perm .= 1:nₚ
    particles
end

# Freeing each gather temporary eagerly lets the device pool hand the same
# blocks to the next component instead of growing until the GC catches up.
# Backends with an eager free (the CUDA and Metal extensions) override the
# hook; elsewhere the temporary is left to the GC.
function permute_through_temporary!(a::AbstractVector, perm)
    tmp = a[perm]
    copyto!(a, tmp)
    free_temporary!(tmp)
    a
end

free_temporary!(::AbstractArray) = nothing
