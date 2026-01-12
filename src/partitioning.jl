const BLOCK_SIZE_LOG2 = unsigned(Preferences.@load_preference("block_size_log2", 2)) # 2^n

abstract type ColoringStrategy end

struct BlockStrategy{dim, Mesh <: CartesianMesh{dim}} <: ColoringStrategy
    mesh::Mesh
    particleindices::Vector{Int}
    stops::Array{Int, dim}
    nparticles_chunks::Vector{Array{Int, dim}}
    blockindices::Vector{Int}
    localindices::Vector{Int}
end

function BlockStrategy(mesh::CartesianMesh{dim}) where {dim}
    blkdims = nblocks(mesh)
    nparticles_chunks = [zeros(Int, blkdims) for _ in 1:Threads.nthreads()]
    particleindices = Int[]
    stops = zeros(Int, blkdims)
    BlockStrategy{dim, typeof(mesh)}(mesh, particleindices, stops, nparticles_chunks, Int[], Int[])
end

nblocks(bs::BlockStrategy) = size(bs.stops)

@inline function particle_indices_in(bs::BlockStrategy, blk::Integer)
    @boundscheck checkbounds(LinearIndices(nblocks(bs)), blk)
    @inbounds _particle_indices_in(bs.particleindices, bs.stops, blk)
end
@inline function particle_indices_in(bs::BlockStrategy, blk::CartesianIndex)
    @boundscheck checkbounds(CartesianIndices(nblocks(bs)), blk)
    @inbounds particle_indices_in(bs, LinearIndices(nblocks(bs))[blk])
end
@inline function _particle_indices_in(particleindices, stops, blk::Integer)
    @_propagate_inbounds_meta
    stop = stops[blk]
    start = blk==1 ? 1 : stops[blk-1]+1
    view(particleindices, start:stop)
end

function update!(bs::BlockStrategy, xₚ::AbstractVector{<: Vec})
    n = length(xₚ)
    resize!(bs.particleindices, n)
    resize!(bs.blockindices, n)
    resize!(bs.localindices, n)
    foreach(fillzero!, bs.nparticles_chunks)

    nchunks = length(bs.nparticles_chunks)
    chunksize = max(1, cld(n, nchunks))
    chunks = [((i-1)*chunksize + 1) : min(i*chunksize, n) for i in 1:nchunks]

    @threaded for chunk_id in 1:nchunks
        @inbounds for p in chunks[chunk_id]
            blk = sub2ind(nblocks(bs), whichblock(xₚ[p], bs.mesh))
            bs.blockindices[p] = blk
            if !iszero(blk)
                bs.localindices[p] = (bs.nparticles_chunks[chunk_id][blk] += 1)
            end
        end
    end
    for i in 1:nchunks-1
        broadcast!(+, bs.nparticles_chunks[i+1], bs.nparticles_chunks[i+1], bs.nparticles_chunks[i])
    end
    nptsinblks = last(bs.nparticles_chunks) # last entry has a complete list

    cumsum!(vec(bs.stops), vec(nptsinblks))
    @threaded for chunk_id in 1:nchunks
        @inbounds for p in chunks[chunk_id]
            blk = bs.blockindices[p]
            if !iszero(blk)
                offset = chunk_id==1 ? 0 : bs.nparticles_chunks[chunk_id-1][blk]
                i = offset + bs.localindices[p]
                stop = bs.stops[blk]
                len = nptsinblks[blk]
                bs.particleindices[stop-len+i] = p
            end
        end
    end

    bs
end
@inline sub2ind(dims::Dims, I)::Int = @inbounds LinearIndices(dims)[I]
@inline sub2ind(::Dims, ::Nothing)::Int = 0

function colorgroups(bs::BlockStrategy)
    [filter(I -> !isempty(particle_indices_in(bs, I)), blocks) for blocks in threadsafe_blocks(nblocks(bs))]
end

function reorder_particles!(particles::AbstractVector, bs::BlockStrategy)
    # NOTE: Only `particleindices` is synced; `blockindices/localindices` are untouched.
    perm = _reorder_particles!(particles, maparray(i -> particle_indices_in(bs, i), LinearIndices(nblocks(bs))))
    invp = invperm(perm)
    n_assigned = bs.stops[end]
    @inbounds for k in 1:n_assigned
        p_old = bs.particleindices[k]
        bs.particleindices[k] = invp[p_old]
    end
    particles
end

function reorder_particles!(particles::AbstractVector, ptsinblks::AbstractArray{<: AbstractVector{Int}})
    _reorder_particles!(particles, ptsinblks)
    particles
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

    # keep missing particles aside
    if nₚ_assigned != nₚ
        @warn "reorder_particles!: Some particles are outside of the grid and were not assigned to any block. They will be kept at the end of the array." maxlog=1
        k = nₚ_assigned
        @inbounds for p in 1:nₚ
            if !seen[p]
                k += 1
                perm[k] = p
            end
        end
        @assert k == nₚ # check just in case
    end

    # reorder particles
    particles_copied = @inbounds particles[perm] # checked in `seen`
    copyto!(particles, 1, particles_copied, 1, nₚ)

    perm
end

####################
# block operations #
####################

nblocks(gridsize::Tuple{Vararg{Int}}) = @. (gridsize-1)>>BLOCK_SIZE_LOG2+1
nblocks(A::AbstractArray) = nblocks(size(A))

@inline function _nodes_in_block(blk::CartesianIndex{dim}) where {dim}
    ranges = ntuple(d -> begin
        i0 = ((blk[d] - 1) << BLOCK_SIZE_LOG2) + 1
        i1 = ( blk[d]      << BLOCK_SIZE_LOG2) + 1
        i0:i1
    end, Val(dim))
    CartesianIndices(ranges)
end
@inline function nodes_in_block(blk::CartesianIndex{dim}, gridsize::Dims{dim}) where {dim}
    nodes = _nodes_in_block(blk) ∩ CartesianIndices(gridsize)
    isempty(nodes) && throw(BoundsError(CartesianIndices(nblocks(gridsize)), Tuple(blk)))
    nodes
end

"""
    Tesserae.whichblock(x::Vec, mesh::CartesianMesh)

Return block index where `x` locates.
The unit block size is `2^$BLOCK_SIZE_LOG2` cells.

# Examples
```jldoctest
julia> mesh = CartesianMesh(1, (0,10), (0,10))
11×11 CartesianMesh{2, Float64, Vector{Float64}}:
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

julia> Tesserae.whichblock(Vec(8.5, 1.5), mesh)
CartesianIndex(3, 1)
```
"""
@inline function whichblock(x::Vec, mesh::CartesianMesh)
    I = findcell(x, mesh)
    I === nothing && return nothing
    CartesianIndex(@. (I.I-1) >> BLOCK_SIZE_LOG2 + 1)
end

function threadsafe_blocks(nblocks::NTuple{dim, Int}) where {dim}
    starts = collect(Iterators.product(ntuple(i->1:2, Val(dim))...))
    vec(map(st -> map(CartesianIndex{dim}, Iterators.product(StepRange.(st, 2, nblocks)...)), starts))
end

struct CellStrategy <: ColoringStrategy
    colorgroups::Vector{Vector{Int}}
end

colorgroups(cs::CellStrategy) = cs.colorgroups

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
    ColorPartition(::CartesianMesh)
    ColorPartition(::UnstructuredMesh)

`ColorPartition` stores partitioning information used by the [`@P2G`](@ref), [`@G2P2G`](@ref) and [`@P2G_Matrix`](@ref) macros
to avoid write conflicts during parallel particle-to-grid transfers.

!!! note
    The [`@threaded`](@ref) macro must be placed before [`@P2G`](@ref), [`@G2P2G`](@ref) and [`@P2G_Matrix`](@ref) to enable parallel transfer.

# Examples
```julia
# Construct ColorPartition
partition = ColorPartition(mesh)

# Update coloring using current particle positions
update!(partition, particles.x) # Required for `CartesianMesh`; not needed for `UnstructuredMesh` (FEM).

# P2G transfer
@threaded @P2G grid=>i particles=>p weights=>ip partition begin
    m[i]  = @∑ w[ip] * m[p]
    mv[i] = @∑ w[ip] * m[p] * v[p]
end
```
"""
struct ColorPartition{Strategy <: ColoringStrategy}
    strategy::Strategy
end

strategy(partition::ColorPartition) = partition.strategy

ColorPartition(mesh::CartesianMesh) = ColorPartition(BlockStrategy(mesh))
ColorPartition(mesh::UnstructuredMesh) = ColorPartition(CellStrategy(mesh))
update!(partition::ColorPartition, args...) = update!(strategy(partition), args...)

reorder_particles!(particles::StructVector, partition::ColorPartition{<: BlockStrategy}) = reorder_particles!(particles, strategy(partition))
