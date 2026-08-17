# -----------------------------------------------------------------------------
#  GPUBlockStrategy
# -----------------------------------------------------------------------------

# GPU sibling of `BlockStrategy`, maintained on the device by a counting sort so
# no host synchronization happens beyond one readback of the active-block count.
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
