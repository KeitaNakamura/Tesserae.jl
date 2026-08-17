# -----------------------------------------------------------------------------
#  Partition strategy core
# -----------------------------------------------------------------------------

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

# Owned by the strategy, so every `partitioned_foreach` entry reuses the same
# group filter and per-group plans instead of allocating them per transfer.
struct RegionScratch{G}
    active::Vector{G}
    bounds::Vector{Vector{Int}}
    cursors::Vector{Threads.Atomic{Int}}
end
RegionScratch{G}() where {G} = RegionScratch{G}(G[], Vector{Int}[], Threads.Atomic{Int}[])

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
