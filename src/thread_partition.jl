abstract type PartitionStrategy end

# Scratch buffers reused by BlockStrategy.update!:
#   * per-chunk block histograms
#   * per-particle packed block ids and 1-based numbers within each chunk/block
#   * touched/active block lists used to avoid full block-array clears/scans
struct BlockUpdateWorkspace{dim}
    chunk_counts::Vector{Array{Int, dim}}
    packed_particle_blocks::Vector{UInt64}
    activeblocks::Vector{Int}
    touchedblocks::Vector{Vector{Int}}
    active_epoch_marks::Vector{Int}
    active_epoch::Base.RefValue{Int}
end

function BlockUpdateWorkspace(blkdims::Dims{dim}) where {dim}
    BlockUpdateWorkspace{dim}(
        [zeros(Int, blkdims) for _ in 1:Threads.nthreads()],
        UInt64[],
        Int[],
        [Int[] for _ in 1:Threads.nthreads()],
        zeros(Int, prod(blkdims)),
        Ref(0),
    )
end

struct BlockStrategy{dim, Mesh <: CartesianMesh{dim}} <: PartitionStrategy
    mesh::Mesh
    particleindices::Vector{Int}
    starts::Array{Int, dim}
    stops::Array{Int, dim}
    nassigned::Base.RefValue{Int}
    activegroups::Vector{Vector{CartesianIndex{dim}}}
    blockcolors::Array{Int, dim}
    update_workspace::BlockUpdateWorkspace{dim}
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

    # Particles are split into fixed chunks by index. For each chunk, count how
    # many particles fall in each block, and remember each particle's block plus
    # its 1-based number within that chunk/block. The scatter pass derives the
    # chunk from the same index ranges, so no per-particle chunk id is needed.
    count_particles_by_block!(bs, xₚ, chunksize, blocklin)

    # Merge per-chunk touched block lists into one unique active block list.
    activate_touched_blocks!(bs)

    # Cache the nonempty color groups used by threaded P2G.
    update_threadsafe_groups!(bs)

    # Assign active block ranges in P2G color-group order.
    build_chunk_offsets_and_ranges!(bs)

    # Scatter particle ids into the block-contiguous index array.
    scatter_particle_indices!(bs, nₚ, chunksize)
    bs
end

function prepare_partition_update!(bs::BlockStrategy, nₚ::Integer)
    ws = bs.update_workspace
    resize!(bs.particleindices, nₚ)
    resize!(ws.packed_particle_blocks, nₚ)
    check_packed_block_number_limits!(bs, nₚ)
    reset_touched_blocks!(bs)

    # Each chunk owns one histogram and touched-block list, so the counting pass
    # can update those buffers without atomics.
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
        touched = ws.touchedblocks[chunk_id]
        firstp = (chunk_id - 1) * chunksize + 1
        lastp = min(chunk_id * chunksize, nₚ)

        @inbounds for p in firstp:lastp
            blk = sub2ind(blocklin, _findblock(xₚ[p], xmin, h_inv, dims, block_size))
            if iszero(blk)
                ws.packed_particle_blocks[p] = 0
            else
                # Record the block once per chunk; activate_touched_blocks!
                # merges these lists without scanning every block.
                count = counts[blk] + 1
                isone(count) && push!(touched, blk)
                counts[blk] = count
                ws.packed_particle_blocks[p] = pack_block_number(blk, count)
            end
        end
    end

    bs
end

function build_chunk_offsets_and_ranges!(bs::BlockStrategy)
    ws = bs.update_workspace
    nchunks = length(ws.chunk_counts)
    blocklin = LinearIndices(nblocks(bs))

    # Turn per-chunk counts into cumulative counts. For a block, counts
    # [4, 2, 5] become [4, 6, 11], so the previous chunk gives the scatter
    # offset and the final chunk gives the block length.
    @inbounds begin
        for chunk_id in 2:nchunks
            counts = ws.chunk_counts[chunk_id]
            prev_counts = ws.chunk_counts[chunk_id - 1]
            for blk in ws.activeblocks
                counts[blk] += prev_counts[blk]
            end
        end

        assigned = 0
        last_counts = ws.chunk_counts[nchunks]
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

# packed_particle_blocks[p] stores two values in one UInt64:
#   upper 32 bits: linear block id
#   lower 32 bits: 1-based number within that particle's chunk/block
# The chunk id is not stored because count and scatter use the same fixed
# particle index ranges.
@inline pack_block_number(block::Integer, number::Integer) =
    (UInt64(block) << PACKED_BLOCK_NUMBER_BITS) | UInt64(number)
@inline packed_block(packed::UInt64) = Int(packed >> PACKED_BLOCK_NUMBER_BITS)
@inline packed_number(packed::UInt64) = Int(packed & PACKED_BLOCK_NUMBER_MASK)

function scatter_particle_indices!(bs::BlockStrategy, nₚ::Integer, chunksize)
    ws = bs.update_workspace

    @threaded for chunk_id in eachindex(ws.chunk_counts)
        firstp = (chunk_id - 1) * chunksize + 1
        lastp = min(chunk_id * chunksize, nₚ)

        @inbounds for p in firstp:lastp
            packed = ws.packed_particle_blocks[p]
            if !iszero(packed)
                blk = packed_block(packed)
                chunk_offset = isone(chunk_id) ? 0 : ws.chunk_counts[chunk_id - 1][blk]
                number = chunk_offset + packed_number(packed)
                # starts[blk] is the block's global range start; number is this
                # particle's 1-based number inside that block after previous chunks.
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

@inline function reset_touched_blocks!(bs::BlockStrategy)
    ws = bs.update_workspace
    # Counts are cumulative after update!, so a block touched by one chunk can
    # leave nonzero entries in later chunks. Clear previous active blocks across
    # all chunks before collecting the next sparse active set.
    @inbounds for counts in ws.chunk_counts
        for blk in ws.activeblocks
            counts[blk] = 0
        end
    end
    for touched in ws.touchedblocks
        empty!(touched)
    end
    @inbounds for blk in ws.activeblocks
        bs.starts[blk] = 0
        bs.stops[blk] = 0
    end
    empty!(ws.activeblocks)
    bs
end

@inline function activate_touched_blocks!(bs::BlockStrategy)
    ws = bs.update_workspace
    epoch = ws.active_epoch[] + 1
    if epoch == typemax(Int)
        fill!(ws.active_epoch_marks, 0)
        epoch = 1
    end
    ws.active_epoch[] = epoch
    marks = ws.active_epoch_marks
    # Epoch marks deduplicate block ids without clearing a full block-sized
    # boolean array every step.
    for touched in ws.touchedblocks
        @inbounds for blk in touched
            if marks[blk] != epoch
                marks[blk] = epoch
                push!(ws.activeblocks, blk)
            end
        end
    end
    ws.activeblocks
end

function update_threadsafe_groups!(bs::BlockStrategy)
    for active in bs.activegroups
        empty!(active)
    end
    cart = CartesianIndices(nblocks(bs))
    @inbounds for blk in bs.update_workspace.activeblocks
        region = cart[blk]
        push!(bs.activegroups[bs.blockcolors[blk]], region)
    end
    bs.activegroups
end
threadsafe_groups(bs::BlockStrategy) = bs.activegroups

function reorder_particles!(particles::AbstractVector, bs::BlockStrategy)
    n_assigned = nassigned(bs)
    _reorder_particles!(particles, bs.particleindices, n_assigned)
    @inbounds for k in 1:n_assigned
        bs.particleindices[k] = k
    end
    particles
end

function reorder_particles!(particles::AbstractVector, ptsinblks::AbstractArray{<: AbstractVector{Int}})
    _reorder_particles!(particles, ptsinblks)
    particles
end

function _reorder_particles!(particles::AbstractVector, particleindices::AbstractVector{Int}, nₚ_assigned::Integer)
    nₚ = length(particles)

    (firstindex(particles) == 1 && lastindex(particles) == nₚ) || throw(ArgumentError("reorder_particles!: particles must be 1-based indexed (`Vector`-like)."))
    nₚ_assigned > nₚ && error("reorder_particles!: The block assignment contains more particle IDs than exist (assigned=$nₚ_assigned, total=$nₚ).")

    perm = Vector{Int}(undef, nₚ)
    !iszero(nₚ_assigned) && copyto!(perm, 1, particleindices, 1, nₚ_assigned)

    if nₚ_assigned != nₚ
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
    end

    particles_copied = @inbounds particles[perm]
    copyto!(particles, 1, particles_copied, 1, nₚ)

    perm
end

function _reorder_particles!(particles::AbstractVector, ptsinblks::AbstractArray{<: AbstractVector{Int}})
    ptsinblks = vec(ptsinblks)
    lens = length.(ptsinblks)
    nₚ = length(particles)
    nₚ_assigned = sum(lens)

    (firstindex(particles) == 1 && lastindex(particles) == nₚ) || throw(ArgumentError("reorder_particles!: particles must be 1-based indexed (`Vector`-like)."))
    nₚ_assigned > nₚ && error("reorder_particles!: The block assignment contains more particle IDs than exist (assigned=$nₚ_assigned, total=$nₚ).")

    offsets = cumsum([0; lens[1:end-1]])
    perm = Vector{Int}(undef, nₚ)
    @threaded for blockindex in eachindex(ptsinblks)
        n = lens[blockindex]
        rng = offsets[blockindex]+1 : offsets[blockindex]+n
        perm[rng] .= ptsinblks[blockindex]
    end

    seen = falses(nₚ)
    for i in 1:nₚ_assigned
        p = perm[i]
        1 ≤ p ≤ nₚ || error("reorder_particles!: particle ID $p is out of range (valid: 1:$nₚ).")
        @inbounds begin
            seen[p] && error("reorder_particles!: particle $p is duplicated in the block assignment.")
            seen[p] = true
        end
    end

    if nₚ_assigned != nₚ
        @warn "reorder_particles!: Some particles are outside of the grid and were not assigned to any block. They will be kept at the end of the array." maxlog=1
        k = nₚ_assigned
        @inbounds for p in 1:nₚ
            if !seen[p]
                k += 1
                perm[k] = p
            end
        end
        @assert k == nₚ
    end

    particles_copied = @inbounds particles[perm]
    copyto!(particles, 1, particles_copied, 1, nₚ)

    perm
end

####################
# block operations #
####################

blockwidth(::Val{L}) where {L} = 1 << L
blockwidth(mesh::CartesianMesh) = blockwidth(Val(block_size_log2(mesh)))

nblocks(gridsize::Tuple{Vararg{Int}}; block_size_log2::Val{L}) where {L} =
    (_check_block_size_log2(block_size_log2); map(n -> ((n - 1) >> L) + 1, gridsize))
nblocks(mesh::CartesianMesh) = nblocks(size(mesh); block_size_log2=Val(block_size_log2(mesh)))

@inline function _nodeindices_in_block(blk::CartesianIndex{dim}, ::Val{L}) where {dim, L}
    ranges = ntuple(d -> begin
        i0 = ((blk[d] - 1) << L) + 1
        i1 = ( blk[d]      << L) + 1
        i0:i1
    end, Val(dim))
    CartesianIndices(ranges)
end
@inline function nodeindices_in_block(blk::CartesianIndex{dim}, gridsize::Dims{dim}; block_size_log2::Val{L}) where {dim, L}
    _check_block_size_log2(block_size_log2)
    nodes = _nodeindices_in_block(blk, block_size_log2) ∩ CartesianIndices(gridsize)
    isempty(nodes) && throw(BoundsError(CartesianIndices(nblocks(gridsize; block_size_log2)), Tuple(blk)))
    nodes
end
@inline nodeindices_in_block(blk::CartesianIndex{dim}, mesh::CartesianMesh{dim}) where {dim} =
    nodeindices_in_block(blk, size(mesh); block_size_log2=Val(block_size_log2(mesh)))

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

# Same boundary rule as findcell, but return the block index directly.
# cell0_d is the 0-based cell index in direction d. It is converted to a
# 1-based block index by shifting by block_size_log2 and adding 1.
@generated function _findblock(x::Vec{dim}, xmin::Vec{dim}, h_inv, dims::Dims{dim}, ::Val{L}) where {dim, L}
    quote
        @_inline_meta
        @nexprs $dim d -> cell0_d = unsafe_trunc(Int, floor((x[d] - xmin[d]) * h_inv))
        inside = @nall $dim d -> 0 ≤ cell0_d ≤ dims[d] - 2
        inside || return nothing
        CartesianIndex(@ntuple $dim d -> (cell0_d >> $L) + 1)
    end
end

function threadsafe_blocks(nblocks::NTuple{dim, Int}) where {dim}
    starts = collect(Iterators.product(ntuple(i->1:2, Val(dim))...))
    vec(map(st -> map(CartesianIndex{dim}, Iterators.product(StepRange.(st, 2, nblocks)...)), starts))
end

struct CellStrategy <: PartitionStrategy
    threadsafe_groups::Vector{Vector{Int}}
end

threadsafe_groups(cs::CellStrategy) = cs.threadsafe_groups

function CellStrategy(mesh::UnstructuredMesh)
    g = _cell_conflict_graph(mesh)

    coloring = Graphs.degree_greedy_color(g)

    groups = [Int[] for _ in 1:coloring.num_colors]
    @inbounds for cell in 1:ncells(mesh)
        push!(groups[coloring.colors[cell]], cell)
    end

    CellStrategy(groups)
end

function _cell_conflict_graph(mesh::UnstructuredMesh)
    nc = ncells(mesh)
    nn = length(mesh)
    graph = SimpleGraph(nc)

    node2cells = [Int[] for _ in 1:nn]
    @inbounds for cell in 1:nc
        for i in cellnodeindices(mesh, cell)
            push!(node2cells[i], cell)
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

"""
    ThreadPartition(::CartesianMesh)
    ThreadPartition(::UnstructuredMesh)

`ThreadPartition` stores partitioning information used by the [`@P2G`](@ref), [`@G2P2G`](@ref) and [`@P2G_Matrix`](@ref) macros
to avoid write conflicts during threaded particle-to-grid transfers.

!!! note
    The [`@threaded`](@ref) macro must be placed before [`@P2G`](@ref), [`@G2P2G`](@ref) and [`@P2G_Matrix`](@ref) to enable parallel transfer.

# Examples
```julia
# Construct ThreadPartition
partition = ThreadPartition(mesh)

# Update partition using current particle positions
update!(partition, particles.x) # Required for `CartesianMesh`; not needed for `UnstructuredMesh` (FEM).

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
ThreadPartition(mesh::UnstructuredMesh) = ThreadPartition(CellStrategy(mesh))
update!(partition::ThreadPartition, args...) = update!(strategy(partition), args...)

reorder_particles!(particles::StructVector, partition::ThreadPartition{<: BlockStrategy}) = reorder_particles!(particles, strategy(partition))
