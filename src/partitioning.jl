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
    dims = blocksize(mesh)
    nparticles_chunks = [zeros(Int, dims) for _ in 1:Threads.nthreads()]
    particleindices = Int[]
    stops = zeros(Int, dims)
    BlockStrategy{dim, typeof(mesh)}(mesh, particleindices, stops, nparticles_chunks, Int[], Int[])
end

blocksize(bs::BlockStrategy) = size(bs.stops)
blockindices(bs::BlockStrategy) = LinearIndices(blocksize(bs))

@inline function particle_indices_in(bs::BlockStrategy, blk::Integer)
    @boundscheck checkbounds(LinearIndices(blocksize(bs)), blk)
    @inbounds _particle_indices_in(bs.particleindices, bs.stops, blk)
end
@inline function particle_indices_in(bs::BlockStrategy, blk::CartesianIndex)
    @boundscheck checkbounds(CartesianIndices(blocksize(bs)), blk)
    @inbounds particle_indices_in(bs, LinearIndices(blocksize(bs))[blk])
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
    chunks = collect(Iterators.partition(1:n, max(1, n÷nchunks+1)))

    @threaded for chunk_id in 1:nchunks
        @inbounds for p in chunks[chunk_id]
            blk = sub2ind(blocksize(bs), whichblock(xₚ[p], bs.mesh))
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
    [filter(I -> !isempty(particle_indices_in(bs, I)), blocks) for blocks in threadsafe_blocks(blocksize(bs))]
end

function reorder_particles!(particles::AbstractVector, bs::BlockStrategy)
    reorder_particles!(particles, maparray(i -> particle_indices_in(bs, i), blockindices(bs)))
end

function reorder_particles!(particles::AbstractVector, ptsinblks::AbstractArray{<: AbstractVector{Int}})
    ptsinblks = vec(ptsinblks)
    lens = length.(ptsinblks)
    offsets = cumsum([0; lens[1:end-1]])
    perm = Vector{Int}(undef, sum(lens))

    @threaded for blockindex in eachindex(ptsinblks)
        n = lens[blockindex]
        rng = offsets[blockindex]+1 : offsets[blockindex]+n
        perm[rng] .= ptsinblks[blockindex]
    end

    # keep missing particles aside
    if length(perm) != length(particles) # some points are missing
        @warn "reorder_particles!: Some particles are outside of the grid and were not assigned to any block. They will be kept at the end of the array." maxlog=1
        missed = particles[setdiff(eachindex(particles), perm)]
    end

    # reorder particles
    @inbounds copyto!(particles, 1, particles[perm], 1, length(perm))

    # assign missing particles to the end part of `particles`
    if length(perm) != length(particles)
        @inbounds particles[length(perm)+1:end] .= missed
    end

    particles
end

####################
# block operations #
####################

blocksize(gridsize::Tuple{Vararg{Int}}) = @. (gridsize-1)>>BLOCK_SIZE_LOG2+1
blocksize(A::AbstractArray) = blocksize(size(A))

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
    I = whichcell(x, mesh)
    I === nothing && return nothing
    CartesianIndex(@. (I.I-1) >> BLOCK_SIZE_LOG2 + 1)
end

function threadsafe_blocks(blocksize::NTuple{dim, Int}) where {dim}
    starts = collect(Iterators.product(ntuple(i->1:2, Val(dim))...))
    vec(map(st -> map(CartesianIndex{dim}, Iterators.product(StepRange.(st, 2, blocksize)...)), starts))
end

"""
    ColorPartition(::CartesianMesh)

`ColorPartition` stores partitioning information used by the [`@P2G`](@ref) and [`@G2P2G`](@ref) macros
to avoid write conflicts during parallel particle-to-grid transfers.

!!! note
    The [`@threaded`](@ref) macro must be placed before [`@P2G`](@ref) and [`@G2P2G`](@ref) to enable parallel transfer.

# Examples
```julia
# Construct ColorPartition
partition = ColorPartition(mesh)

# Update coloring using current particle positions
update!(partition, particles.x)

# P2G transfer
@threaded @P2G grid=>i particles=>p mpvalues=>ip partition begin
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
update!(partition::ColorPartition, args...) = update!(strategy(partition), args...)

reorder_particles!(particles::StructVector, partition::ColorPartition{<: BlockStrategy}) = reorder_particles!(particles, strategy(partition))
