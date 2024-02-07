const BLOCKFACTOR = unsigned(Preferences.@load_preference("block_factor", 3)) # 2^n

struct SpSpace{dim, L <: Lattice{dim}, ElType} <: AbstractArray{ElType, dim}
    lattice::L
    particleindices::Vector{Int}
    stops::Array{Int, dim}
    nparticles::Vector{Array{Int, dim}}
    blockindices::Vector{Int}
    localindices::Vector{Int}
end

function SpSpace(lattice::Lattice{dim}) where {dim}
    dims = blocksize(lattice)
    nparticles = [zeros(Int, dims) for _ in 1:Threads.nthreads()]
    particleindices = Int[]
    stops = zeros(Int, dims)
    ElType = Base._return_type(_getindex, Tuple{typeof(particleindices), typeof(stops), Int})
    SpSpace{dim, typeof(lattice), ElType}(lattice, particleindices, stops, nparticles, Int[], Int[])
end

Base.IndexStyle(::Type{<: SpSpace}) = IndexLinear()
Base.size(x::SpSpace) = size(x.stops)
@inline function Base.getindex(x::SpSpace, i::Integer)
    @boundscheck checkbounds(x, i)
    @inbounds _getindex(x.particleindices, x.stops, i)
end
@inline function _getindex(particleindices, stops, i)
    @_propagate_inbounds_meta
    stop = stops[i]
    start = i==1 ? 1 : stops[i-1]+1
    view(particleindices, start:stop)
end

function update!(s::SpSpace, xₚ::AbstractVector{<: Vec})
    n = length(xₚ)
    resize!(s.particleindices, n)
    resize!(s.blockindices, n)
    resize!(s.localindices, n)
    foreach(fillzero!, s.nparticles)

    @threaded :static for p in 1:n
        id = Threads.threadid()
        blk = sub2ind(size(s), whichblock(xₚ[p], s.lattice))
        s.blockindices[p] = blk
        s.localindices[p] = iszero(blk) ? 0 : (s.nparticles[id][blk] += 1)
    end
    for i in 1:Threads.nthreads()-1
        broadcast!(+, s.nparticles[i+1], s.nparticles[i+1], s.nparticles[i])
    end
    nptsinblks = last(s.nparticles) # last entry has a complete list

    cumsum!(vec(s.stops), vec(nptsinblks))
    @threaded :static for p in 1:n
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

    s
end
sub2ind(dims::Dims, I)::Int = @inbounds LinearIndices(dims)[I]
sub2ind(::Dims, ::Nothing)::Int = 0

function update_block_sparsity!(spinds::SpIndices, s::SpSpace)
    blocksize(spinds) == size(s) || throw(ArgumentError("block size $(blocksize(spinds)) must match"))

    inds = fillzero!(blockindices(spinds))
    CI = CartesianIndices(s)
    @inbounds for I in CI
        if !isempty(s[I])
            blks = (I - oneunit(I)):(I + oneunit(I))
            inds[blks ∩ CI] .= true
        end
    end

    numbering!(spinds)
end

function threadsafe_blocks(s::SpSpace)
    [filter(I -> !isempty(s[I]), blocks) for blocks in threadsafe_blocks(size(s))]
end

function reorder_particles!(particles::AbstractVector, ptsinblks::AbstractArray{<: AbstractVector{Int}})
    inds = Vector{Int}(undef, sum(length, ptsinblks))

    cnt = 1
    for blocks in threadsafe_blocks(size(ptsinblks))
        @inbounds for blockindex in blocks
            particleindices = ptsinblks[blockindex]
            for i in eachindex(particleindices)
                inds[cnt] = particleindices[i]
                particleindices[i] = cnt
                cnt += 1
            end
        end
    end

    # keep missing particles aside
    if length(inds) != length(particles) # some points are missing
        missed = particles[setdiff(1:length(particles), inds)]
    end

    # reorder particles
    @inbounds particles[1:length(inds)] .= view(particles, inds)

    # assign missing particles to the end part of `particles`
    if length(inds) != length(particles)
        @inbounds particles[length(inds)+1:end] .= missed
    end

    particles
end

####################
# block operations #
####################

blocksize(gridsize::Tuple{Vararg{Int}}) = @. (gridsize-1)>>BLOCKFACTOR+1
blocksize(A::AbstractArray) = blocksize(size(A))

"""
    Sequoia.whichblock(x::Vec, lattice::Lattice)

Return block index where `x` locates.
The unit block size is `2^$BLOCKFACTOR` cells.

# Examples
```jldoctest
julia> lattice = Lattice(1, (0,10), (0,10))
11×11 Lattice{2, Float64, Vector{Float64}}:
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

julia> Sequoia.whichblock(Vec(8.5, 1.5), lattice)
CartesianIndex(2, 1)
```
"""
@inline function whichblock(x::Vec, lattice::Lattice)
    I = whichcell(x, lattice)
    I === nothing && return nothing
    CartesianIndex(@. (I.I-1) >> BLOCKFACTOR + 1)
end

function threadsafe_blocks(blocksize::NTuple{dim, Int}) where {dim}
    starts = AxisArray(nfill(1:2, Val(dim)))
    vec(map(st -> map(CartesianIndex{dim}, AxisArray(StepRange.(st, 2, blocksize)))::Array{CartesianIndex{dim}, dim}, starts))
end
