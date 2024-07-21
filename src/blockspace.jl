const BLOCKFACTOR = unsigned(Preferences.@load_preference("block_factor", 3)) # 2^n

struct BlockSpace{dim, L <: CartesianMesh{dim}, ElType} <: AbstractArray{ElType, dim}
    mesh::L
    particleindices::Vector{Int}
    stops::Array{Int, dim}
    nparticles::Vector{Array{Int, dim}}
    blockindices::Vector{Int}
    localindices::Vector{Int}
end

function BlockSpace(mesh::CartesianMesh{dim}) where {dim}
    dims = blocksize(mesh)
    nparticles = [zeros(Int, dims) for _ in 1:Threads.nthreads()]
    particleindices = Int[]
    stops = zeros(Int, dims)
    ElType = Base._return_type(_getindex, Tuple{typeof(particleindices), typeof(stops), Int})
    BlockSpace{dim, typeof(mesh), ElType}(mesh, particleindices, stops, nparticles, Int[], Int[])
end

Base.IndexStyle(::Type{<: BlockSpace}) = IndexLinear()
Base.size(x::BlockSpace) = size(x.stops)
@inline function Base.getindex(x::BlockSpace, i::Integer)
    @boundscheck checkbounds(x, i)
    @inbounds _getindex(x.particleindices, x.stops, i)
end
@inline function _getindex(particleindices, stops, i)
    @_propagate_inbounds_meta
    stop = stops[i]
    start = i==1 ? 1 : stops[i-1]+1
    view(particleindices, start:stop)
end

function update!(s::BlockSpace, xₚ::AbstractVector{<: Vec})
    n = length(xₚ)
    resize!(s.particleindices, n)
    resize!(s.blockindices, n)
    resize!(s.localindices, n)
    foreach(fillzero!, s.nparticles)

    @threaded :static for p in 1:n
        @inbounds begin
            id = Threads.threadid()
            blk = sub2ind(size(s), whichblock(xₚ[p], s.mesh))
            s.blockindices[p] = blk
            if !iszero(blk)
                s.localindices[p] = (s.nparticles[id][blk] += 1)
            end
        end
    end
    for i in 1:Threads.nthreads()-1
        broadcast!(+, s.nparticles[i+1], s.nparticles[i+1], s.nparticles[i])
    end
    nptsinblks = last(s.nparticles) # last entry has a complete list

    cumsum!(vec(s.stops), vec(nptsinblks))
    @threaded :static for p in 1:n
        @inbounds begin
            blk = s.blockindices[p]
            if !iszero(blk)
                id = Threads.threadid()
                offset = id==1 ? 0 : s.nparticles[id-1][blk]
                i = offset + s.localindices[p]
                stop = s.stops[blk]
                len = nptsinblks[blk]
                s.particleindices[stop-len+i] = p
            end
        end
    end

    s
end
@inline sub2ind(dims::Dims, I)::Int = @inbounds LinearIndices(dims)[I]
@inline sub2ind(::Dims, ::Nothing)::Int = 0

function threadsafe_blocks(s::BlockSpace)
    [filter(I -> !isempty(s[I]), blocks) for blocks in threadsafe_blocks(size(s))]
end

function reorder_particles!(particles::AbstractVector, ptsinblks::AbstractArray{<: AbstractVector{Int}})
    perm = Vector{Int}(undef, sum(length, ptsinblks))

    count = Threads.Atomic{Int}(1)
    for blocks in threadsafe_blocks(size(ptsinblks))
        @threaded for blockindex in blocks
            particleindices = ptsinblks[blockindex]
            n = length(particleindices)
            cnt = Threads.atomic_add!(count, n)
            rng = cnt:cnt+n-1
            perm[rng] .= particleindices
            particleindices .= rng
        end
    end

    # keep missing particles aside
    if length(perm) != length(particles) # some points are missing
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

blocksize(gridsize::Tuple{Vararg{Int}}) = @. (gridsize-1)>>BLOCKFACTOR+1
blocksize(A::AbstractArray) = blocksize(size(A))

"""
    Tesserae.whichblock(x::Vec, mesh::CartesianMesh)

Return block index where `x` locates.
The unit block size is `2^$BLOCKFACTOR` cells.

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
CartesianIndex(2, 1)
```
"""
@inline function whichblock(x::Vec, mesh::CartesianMesh)
    I = whichcell(x, mesh)
    I === nothing && return nothing
    CartesianIndex(@. (I.I-1) >> BLOCKFACTOR + 1)
end

function threadsafe_blocks(blocksize::NTuple{dim, Int}) where {dim}
    starts = collect(Iterators.product(ntuple(i->1:2, Val(dim))...))
    vec(map(st -> map(CartesianIndex{dim}, Iterators.product(StepRange.(st, 2, blocksize)...)), starts))
end
