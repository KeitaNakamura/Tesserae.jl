# -----------------------------------------------------------------------------
#  Partition strategy core
# -----------------------------------------------------------------------------

abstract type PartitionStrategy end

# The CPU and GPU block strategies partition the same mesh blocks and share the
# accessor contract; only how they maintain the assignment differs.
abstract type BlockStrategy <: PartitionStrategy end

nblocks(bs::BlockStrategy) = nblocks(bs.mesh)
block_size_log2(bs::BlockStrategy) = block_size_log2(bs.mesh)
blockwidth(bs::BlockStrategy) = blockwidth(bs.mesh)
nassigned(bs::BlockStrategy) = bs.nassigned[]

# Owned by the strategy, so every `partitioned_foreach` entry reuses the same
# group filter and per-group plans instead of allocating them per transfer.
struct RegionScratch{G}
    active::Vector{G}
    bounds::Vector{Vector{Int}}
    cursors::Vector{Threads.Atomic{Int}}
end
RegionScratch{G}() where {G} = RegionScratch{G}(G[], Vector{Int}[], Threads.Atomic{Int}[])

struct BlockMatrixBufferPool
    lock::ReentrantLock
    buffers::Dict{Any, Vector{Any}}
end
BlockMatrixBufferPool() = BlockMatrixBufferPool(ReentrantLock(), Dict{Any, Vector{Any}}())
