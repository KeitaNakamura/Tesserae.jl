struct SpIndex{I}
    index::I
    spindex::Int
end

# `SpIndex` is tied to the current numbering of a `SpIndices` object.
# It should be treated as a short-lived token and not stored across
# `update_sparsity!` calls.
@inline logicalindex(x::SpIndex) = x.index
@inline storageindex(x::SpIndex) = x.spindex
isactive(x::SpIndex) = !iszero(x.spindex)

@inline elone(A) = one(eltype(A))
@inline elzero(A) = zero(eltype(A))

@inline function Base.getindex(A::AbstractArray, i::SpIndex)
    @boundscheck checkbounds(A, logicalindex(i))
    @inbounds isactive(i) ? A[logicalindex(i)] : zero_recursive(eltype(A))
end

Base.show(io::IO, x::SpIndex) = print(io, "SpIndex(", x.index, ", ", ifelse(isactive(x), x.spindex, CDot()), ")")

function resize_fillzero!(A::AbstractVector, n::Integer)
    # Existing Metal buffers cannot resize to zero length. Zeroing stale
    # storage is enough when the active numbering no longer references it.
    if iszero(n) && get_device(A) isa MetalDevice
        fillzero!(A)
    else
        fillzero!(resize!(A, n))
    end
    A
end

struct ParticleBlockTracker{B <: AbstractArray{Int}, C <: AbstractArray{Int32}}
    blockids::B  # block id currently recorded for each particle
    counts::C    # number of particles currently recorded in each block
end

function ParticleBlockTracker(blocknumbering::AbstractArray)
    blockids = similar(vec(blocknumbering), Int, 0)
    counts = fillzero!(similar(blocknumbering, Int32))
    ParticleBlockTracker(blockids, counts)
end

# Owned by `SpIndices`, so particle-driven sparsity updates reuse storage
# instead of allocating block-sized temporaries every step.
struct BlockSparsityWorkspace{O <: AbstractArray{Bool}, A <: AbstractArray{Bool}, T <: ParticleBlockTracker}
    occupied::O  # blocks containing particles
    active::A    # blocks allocated for basis support
    tracker::T   # particle block ids and per-block particle counts
end

function BlockSparsityWorkspace(blocknumbering::AbstractArray)
    occupied = fillzero!(similar(blocknumbering, Bool))
    active = fillzero!(similar(blocknumbering, Bool))
    BlockSparsityWorkspace(occupied, active, ParticleBlockTracker(blocknumbering))
end

# Block sparsity is stored as a dense array over block coordinates.
# Zero means inactive; positive values are compact blocknumbers for SpArray.data.
struct SpIndices{dim, L, B <: AbstractArray{Int, dim}, W <: BlockSparsityWorkspace} <: AbstractArray{SpIndex{CartesianIndex{dim}}, dim}
    dims::Dims{dim}
    blocknumbering::B
    workspace::W
end

function SpIndices(dims::Dims{dim}; block_size_log2::Val{L}=Val(BLOCK_SIZE_LOG2)) where {dim, L}
    _check_block_size_log2(block_size_log2)
    blocknumbering = fill(0, nblocks(dims; block_size_log2))
    workspace = BlockSparsityWorkspace(blocknumbering)
    SpIndices{dim, L, typeof(blocknumbering), typeof(workspace)}(dims, blocknumbering, workspace)
end
SpIndices(dims::Int...; kwargs...) = SpIndices(dims; kwargs...)
SpIndices(mesh::CartesianMesh) = SpIndices(size(mesh); block_size_log2=Val(block_size_log2(mesh)))

Base.size(sp::SpIndices) = sp.dims
Base.IndexStyle(::Type{<: SpIndices}) = IndexCartesian()

@inline blocknumbering(sp::SpIndices) = sp.blocknumbering
@inline sparsity_workspace(sp::SpIndices) = sp.workspace
@inline occupied_blocks(sp::SpIndices) = sparsity_workspace(sp).occupied
@inline active_blocks(sp::SpIndices) = sparsity_workspace(sp).active
@inline sparsity_tracker(sp::SpIndices) = sparsity_workspace(sp).tracker
@inline nblocks(sp::SpIndices) = size(blocknumbering(sp))
@inline block_size_log2(::SpIndices{dim, L}) where {dim, L} = L

# Each active block stores a dense block of size blocksize(sp) in SpArray.data.
@inline blockwidth(sp::SpIndices) = blockwidth(Val(block_size_log2(sp)))
@inline blocksize(sp::SpIndices{dim}) where {dim} = nfill(blockwidth(sp), Val(dim))
@inline blocklength(sp::SpIndices{dim, L}) where {dim, L} = 1 << (L*dim)

# blocknumber + local linear index inside the block -> SpArray.data index.
@inline storageindex(blocknumber::Integer, localindex::Integer, sp::SpIndices) =
    (blocknumber - 1) * blocklength(sp) + localindex

# Logical node index -> block coordinate.
@inline blockindex(I::Vararg{Integer, dim}; block_size_log2::Val{L}) where {dim, L} =
    @. ((I - 1) >> L) + 1

# Logical node index -> block coordinate and local linear index inside the block.
@inline function global_to_blocklocal(I::Vararg{Integer, dim}; block_size_log2::Val{L}) where {dim, L}
    j = I .- 1
    block = blockindex(I...; block_size_log2)
    localcoord = @. (j & ((1 << L) - 1)) + 1
    LI = LinearIndices(nfill(1 << L, Val(dim)))
    @inbounds block, LI[localcoord...]
end

# block coordinate and local Cartesian index inside the block -> logical node index.
@inline function blocklocal_to_global(block::CartesianIndex{dim}, localcoord::CartesianIndex{dim}; block_size_log2::Val{L}) where {dim, L}
    CartesianIndex(ntuple(d -> ((block[d] - 1) << L) + localcoord[d], Val(dim)))
end

# GPU kernels cannot iterate `activeindices(spinds)` directly, so they launch over
# block-local slots and recover the corresponding active `SpIndex` with these helpers.
@inline _spindex_ndrange(spinds::SpIndices) = length(blocknumbering(spinds)) * blocklength(spinds)

@inline function _active_spindex(spinds::SpIndices, blocknumber, block::CartesianIndex, l::Integer, localindices)
    iszero(blocknumber) && return false, SpIndex(block, 0)
    @inbounds localcoord = localindices[l]
    I = blocklocal_to_global(block, localcoord; block_size_log2=Val(block_size_log2(spinds)))
    checkbounds(Bool, spinds, Tuple(I)...) || return false, SpIndex(I, 0)
    true, SpIndex(I, storageindex(blocknumber, l, spinds))
end

@inline function _active_spindex(spinds::SpIndices, k::Integer)
    numbering = blocknumbering(spinds)
    blocks = CartesianIndices(numbering)
    localindices = CartesianIndices(blocksize(spinds))
    nlocal = length(localindices)
    b = (k - 1) ÷ nlocal + 1
    l = (k - 1) % nlocal + 1
    @inbounds blocknumber = numbering[b]
    @inbounds block = blocks[b]
    _active_spindex(spinds, blocknumber, block, l, localindices)
end

@inline function Base.getindex(sp::SpIndices{dim}, I::Vararg{Integer, dim}) where {dim}
    @boundscheck checkbounds(sp, I...)
    block_size = Val(block_size_log2(sp))
    block, localindex = global_to_blocklocal(I...; block_size_log2=block_size)
    @inbounds blocknumber = blocknumbering(sp)[block...]
    index = storageindex(blocknumber, localindex, sp)
    SpIndex(CartesianIndex(I), ifelse(iszero(blocknumber), zero(index), index))
end

struct ActiveSpIndices{dim, S <: SpIndices{dim}}
    spinds::S
end

# Iterate active logical indices in storage order. This is intentionally not
# Cartesian iteration order; callers that work with `SpArray.data` can use the
# resulting `SpIndex` values without re-sorting.
activeindices(sp::SpIndices) = ActiveSpIndices(sp)

Base.IteratorSize(::Type{<: ActiveSpIndices}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<: ActiveSpIndices}) = Base.HasEltype()
Base.eltype(::Type{ActiveSpIndices{dim, S}}) where {dim, S} = SpIndex{CartesianIndex{dim}}

function Base.iterate(iter::ActiveSpIndices{dim}, state=(1, 1)) where {dim}
    sp = iter.spinds
    numbering = blocknumbering(sp)
    blocks = CartesianIndices(numbering)
    localindices = CartesianIndices(blocksize(sp))
    block_size = Val(block_size_log2(sp))
    nblock = length(numbering)
    nlocal = length(localindices)
    b, l = state

    @inbounds while b ≤ nblock
        blocknumber = numbering[b]
        if !iszero(blocknumber)
            block = blocks[b]
            while l ≤ nlocal
                localcoord = localindices[l]
                I = blocklocal_to_global(block, localcoord; block_size_log2=block_size)
                i = storageindex(blocknumber, l, sp)
                l += 1
                checkbounds(Bool, sp, Tuple(I)...) && return SpIndex(I, i), (b, l)
            end
        end
        b += 1
        l = 1
    end

    nothing
end

@inline function isactive(sp::SpIndices{dim}, I::Vararg{Integer, dim}) where {dim}
    @boundscheck checkbounds(sp, I...)
    block = blockindex(I...; block_size_log2=Val(block_size_log2(sp)))
    @inbounds !iszero(blocknumbering(sp)[block...])
end
@inline isactive(sp::SpIndices, I::CartesianIndex) = (@_propagate_inbounds_meta; isactive(sp, Tuple(I)...))

function _check_nblocks(sp::SpIndices, blocks)
    nblocks(sp) == size(blocks) || throw(ArgumentError("blocks per dimension $(nblocks(sp)) must match"))
end
function _check_nblocks(sp::SpIndices, mesh::CartesianMesh)
    nblocks(sp) == nblocks(mesh) || throw(ArgumentError("blocks per dimension $(nblocks(sp)) must match"))
end

function _check_same_backend(label, x, backend)
    get_backend(x) == backend || throw(ArgumentError("SpIndices and $label must live on the same backend"))
end

function update_sparsity!(sp::SpIndices, blkspy::AbstractArray)
    _apply_block_activity!(sp, blkspy)
end

function update_sparsity!(spinds::SpIndices, xₚ::AbstractVector{<: Vec}, mesh::CartesianMesh)
    backend = get_backend(blocknumbering(spinds))
    _check_same_backend("particle positions", xₚ, backend)
    _check_same_backend("mesh", mesh, backend)
    _check_nblocks(spinds, mesh)

    if isempty(xₚ)
        _reset_particle_block_tracker!(spinds, 0)
        return _apply_block_activity!(spinds, fillzero!(active_blocks(spinds)); preserve_tracker=true)
    end

    # Reuse numbering when the occupied block set is unchanged.
    if !_update_particle_block_tracker!(spinds, xₚ, mesh, backend)
        return nothing
    end
    activity = _activate_neighbor_blocks!(active_blocks(spinds), occupied_blocks(spinds), backend)
    _apply_block_activity!(spinds, activity; preserve_tracker=true)
end

# Block activity -> compact block numbering.
function _apply_block_activity!(sp::SpIndices, activity; preserve_tracker::Bool=false)
    _check_nblocks(sp, activity)
    _check_same_backend("block activity", activity, get_backend(blocknumbering(sp)))
    n = _number_blocks!(sp, activity)
    preserve_tracker || _invalidate_particle_block_tracker!(sp)
    n
end

_number_blocks!(sp::SpIndices, activity) = _number_blocks!(sp, activity, get_backend(blocknumbering(sp)))

function _number_blocks!(sp::SpIndices, activity, ::CPU)
    numbers = blocknumbering(sp)
    active_block_count = 0
    @inbounds for i in eachindex(numbers, activity)
        numbers[i] = iszero(activity[i]) ? 0 : (active_block_count += 1)
    end
    active_block_count * blocklength(sp)
end

@kernel function gpukernel_init_block_numbering!(block_numbers, @Const(activity))
    b = @index(Global)
    @inbounds block_numbers[b] = ifelse(iszero(activity[b]), elzero(block_numbers), elone(block_numbers))
end

@kernel function gpukernel_finalize_block_numbering!(block_numbers, @Const(activity), active_count)
    b = @index(Global)
    @inbounds if b == length(block_numbers)
        active_count[] = block_numbers[b]
    end
    @inbounds if iszero(activity[b])
        block_numbers[b] = 0
    end
end

function _number_blocks!(sp::SpIndices, activity, backend::GPU)
    block_numbers = blocknumbering(sp)
    active_count_buffer = similar(vec(block_numbers), eltype(block_numbers), 1)

    # Build compact block numbers with an inclusive scan:
    # activity -> 0/1 markers -> prefix sum -> inactive blocks reset to 0.
    init_kernel = gpukernel_init_block_numbering!(backend)
    init_kernel(block_numbers, activity; ndrange=length(block_numbers))

    cumsum!(vec(block_numbers), vec(block_numbers))

    finalize_kernel = gpukernel_finalize_block_numbering!(backend)
    finalize_kernel(block_numbers, activity, active_count_buffer; ndrange=length(block_numbers))
    # Only sync before the CPU reads `active_count_buffer`.
    synchronize(backend)
    only(Array(active_count_buffer)) * blocklength(sp)
end

# Particle block tracker.
# Manual sparsity updates bypass particle positions, so the particle tracker is
# no longer a valid description of the current sparsity state.
function _invalidate_particle_block_tracker!(spinds::SpIndices)
    tracker = sparsity_tracker(spinds)
    resize_fillzero!(tracker.blockids, 0)
    fillzero!(tracker.counts)
    fillzero!(occupied_blocks(spinds))
    spinds
end

# Rebuild tracker storage on the first particle update, after invalidation, or
# when the particle count changes.
function _reset_particle_block_tracker!(spinds::SpIndices, nparticles::Integer)
    tracker = sparsity_tracker(spinds)
    length(tracker.blockids) == nparticles && return false
    resize_fillzero!(tracker.blockids, nparticles)
    fillzero!(tracker.counts)
    fillzero!(occupied_blocks(spinds))
    true
end

@inline blockid(dims::Dims, x::Vec, mesh::CartesianMesh)::Int = sub2ind(dims, findblock(x, mesh))

# Update particle -> block ids and per-block particle counts. The expensive
# active expansion and numbering are needed only when occupied blocks change.
function _update_particle_block_tracker!(spinds::SpIndices, xₚ, mesh, ::CPU)
    reset = _reset_particle_block_tracker!(spinds, length(xₚ))
    tracker = sparsity_tracker(spinds)
    blockids = tracker.blockids
    counts = tracker.counts
    moved = reset

    @inbounds for p in eachindex(xₚ)
        new = blockid(size(counts), xₚ[p], mesh)
        old = blockids[p]
        if old != new
            blockids[p] = new
            moved = true
            iszero(old) || (counts[old] -= elone(counts))
            iszero(new) || (counts[new] += elone(counts))
        end
    end

    moved && _refresh_occupied_blocks!(occupied_blocks(spinds), counts, reset)
end

# Convert block particle counts into occupied flags and report whether the
# occupied set changed.
function _refresh_occupied_blocks!(occupied, counts, tracker_reset::Bool)
    changed = tracker_reset
    @inbounds for i in eachindex(occupied, counts)
        now = !iszero(counts[i])
        if now != !iszero(occupied[i])
            occupied[i] = now
            changed = true
        end
    end
    changed
end

@kernel function gpukernel_update_particle_block_tracker!(blockids, counts, @Const(xₚ), @Const(mesh))
    p = @index(Global)
    new = blockid(size(counts), xₚ[p], mesh)
    @inbounds old = blockids[p]
    if old != new
        @inbounds blockids[p] = new
        if !iszero(old)
            @inbounds Atomix.@atomic counts[old] -= elone(counts)
        end
        if !iszero(new)
            @inbounds Atomix.@atomic counts[new] += elone(counts)
        end
    end
end

@kernel function gpukernel_refresh_occupied_blocks!(occupied_blocks, @Const(counts), changed)
    b = @index(Global)
    @inbounds begin
        now = !iszero(counts[b])
        if now != !iszero(occupied_blocks[b])
            occupied_blocks[b] = now
            Atomix.@atomic changed[] += elone(changed)
        end
    end
end

function _update_particle_block_tracker!(spinds::SpIndices, xₚ, mesh, backend::GPU)
    reset = _reset_particle_block_tracker!(spinds, length(xₚ))
    tracker = sparsity_tracker(spinds)
    # Only the occupied set decides whether active expansion and numbering are
    # needed; individual particle moves are just an intermediate detail.
    changed = fillzero!(similar(vec(blocknumbering(spinds)), Int32, 1))

    update_kernel = gpukernel_update_particle_block_tracker!(backend)
    update_kernel(tracker.blockids, tracker.counts, xₚ, mesh; ndrange=length(xₚ))

    # Launches on the same backend are ordered; only sync before the CPU reads `changed`.
    refresh_kernel = gpukernel_refresh_occupied_blocks!(backend)
    refresh_kernel(occupied_blocks(spinds), tracker.counts, changed; ndrange=length(occupied_blocks(spinds)))
    synchronize(backend)
    reset || !iszero(only(Array(changed)))
end

# Occupied blocks -> active blocks for basis support.
function _activate_block_neighborhood!(active_blocks, I::CartesianIndex, CI)
    blks = (I - oneunit(I)):(I + oneunit(I))
    active_blocks[blks ∩ CI] .= true
    active_blocks
end

function _activate_neighbor_blocks!(active, occupied, ::CPU)
    fillzero!(active)
    CI = CartesianIndices(active)
    @inbounds for I in CartesianIndices(occupied)
        iszero(occupied[I]) || _activate_block_neighborhood!(active, I, CI)
    end
    active
end

@inline function _inbounds_block(I::CartesianIndex{dim}, dims::Dims{dim}) where {dim}
    all(ntuple(d -> 1 ≤ I[d] ≤ dims[d], Val(dim)))
end

# GPU particle-driven updates expand occupied blocks here instead of relying on
# CPU ThreadPartition scheduling. Multiple threads may write the same `true`;
# only the final boolean state matters.
@kernel function gpukernel_expand_occupied_blocks!(active_blocks, @Const(occupied_blocks))
    b = @index(Global)
    @inbounds if !iszero(occupied_blocks[b])
        dims = size(occupied_blocks)
        blk = CartesianIndices(dims)[b]
        for offset in CartesianIndices(nfill(-1:1, Val(length(dims))))
            neighbor = CartesianIndex(ntuple(d -> blk[d] + offset[d], Val(length(dims))))
            if _inbounds_block(neighbor, dims)
                active_blocks[sub2ind(dims, neighbor)] = true
            end
        end
    end
end

function _activate_neighbor_blocks!(active, occupied, backend::GPU)
    fillzero!(active)
    expand_kernel = gpukernel_expand_occupied_blocks!(backend)
    expand_kernel(active, occupied; ndrange=length(occupied))
    active
end

function update_sparsity!(spinds::SpIndices{dim, <:Any, <:Array{Int, dim}}, partition::ThreadPartition{<: BlockStrategy}) where {dim}
    bs = strategy(partition)
    nblocks(spinds) == nblocks(bs) || throw(ArgumentError("blocks per dimension $(nblocks(spinds)) must match"))
    block_size_log2(spinds) == block_size_log2(bs) ||
        throw(ArgumentError("block_size_log2 $(block_size_log2(spinds)) must match partition block_size_log2 $(block_size_log2(bs))"))

    activity = fillzero!(active_blocks(spinds))
    CI = CartesianIndices(activity)
    @inbounds for I in CI
        if !isempty(particle_indices(bs, I))
            _activate_block_neighborhood!(activity, I, CI)
        end
    end

    _apply_block_activity!(spinds, activity)
end

"""
    SpArray{T}(undef, dims...)

`SpArray` is a sparse array which has blockwise sparsity pattern.
In `SpArray`, it is not allowed to freely change the value like built-in `Array`.
For example, trying to `setindex!` doesn't change anything without any errors as

```jldoctest sparray
julia> A = SpArray{Float64}(undef, 5, 5)
5×5 SpArray{Float64, 2, Vector{Float64}, Tesserae.SpIndices{2, 2, Matrix{Int64}, Tesserae.BlockSparsityWorkspace{Matrix{Bool}, Matrix{Bool}, Tesserae.ParticleBlockTracker{Vector{Int64}, Matrix{Int32}}}}}:
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅

julia> A[1,1]
0.0

julia> A[1,1] = 2 # no error
2

julia> A[1,1] # still zero
0.0
```

This is because the block where index `(1,1)` is located is not activated yet.
To activate the block, update sparsity pattern by `update_sparsity!(A, spy)`
where `spy` must have `Tesserae.nblocks(A)`.

```jldoctest sparray
julia> spy = trues(Tesserae.nblocks(A))
2×2 BitMatrix:
 1  1
 1  1

julia> update_sparsity!(A, spy) # returned value indicates the number of allocated elements in `A`.
64

julia> A .= 0;

julia> A[1,1] = 2
2

julia> A
5×5 SpArray{Float64, 2, Vector{Float64}, Tesserae.SpIndices{2, 2, Matrix{Int64}, Tesserae.BlockSparsityWorkspace{Matrix{Bool}, Matrix{Bool}, Tesserae.ParticleBlockTracker{Vector{Int64}, Matrix{Int32}}}}}:
 2.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
```
"""
struct SpArray{T, dim, D <: AbstractVector{T}, S <: SpIndices{dim}} <: AbstractArray{T, dim}
    data::D
    spinds::S
    shared_spinds::Bool
end

function SpArray{T}(::UndefInitializer, dims::Tuple{Vararg{Int}}; block_size_log2::Val{L}=Val(BLOCK_SIZE_LOG2)) where {T, L}
    data = Vector{T}(undef, 0)
    spinds = SpIndices(dims; block_size_log2)
    SpArray(data, spinds, false)
end
SpArray{T}(::UndefInitializer, dims::Int...; kwargs...) where {T} = SpArray{T}(undef, dims; kwargs...)

function SpArray{T}(spinds::SpIndices) where {T}
    data = Vector{T}(undef, 0)
    SpArray(data, spinds, true)
end

Base.IndexStyle(::Type{<: SpArray}) = IndexCartesian()
Base.size(A::SpArray) = size(A.spinds)

get_data(A::SpArray) = A.data
get_spinds(A::SpArray) = A.spinds
nblocks(A::SpArray) = nblocks(get_spinds(A))
storedindices(A::SpArray) = eachindex(get_data(A))
activeindices(A::SpArray) = activeindices(get_spinds(A))

function Base.fill!(A::SpArray, x)
    fill!(get_data(A), x)
    A
end

# return zero if the index is not active
@inline function Base.getindex(A::SpArray, i::SpIndex)
    @boundscheck checkbounds(A, logicalindex(i))
    isactive(i) || return zero_recursive(eltype(A))
    @debug checkbounds(get_data(A), storageindex(i))
    @inbounds get_data(A)[storageindex(i)]
end

# do nothing if the index is not active (do not throw error!!)
@inline function Base.setindex!(A::SpArray, v, i::SpIndex)
    @boundscheck checkbounds(A, logicalindex(i))
    isactive(i) || return A
    @debug checkbounds(get_data(A), storageindex(i))
    @inbounds get_data(A)[storageindex(i)] = v
    A
end

@inline function Base.getindex(A::SpArray{<: Any, dim}, I::Vararg{Integer, dim}) where {dim}
    @_propagate_inbounds_meta
    A[get_spinds(A)[I...]]
end

@inline function Base.setindex!(A::SpArray{<: Any, dim}, v, I::Vararg{Integer, dim}) where {dim}
    @_propagate_inbounds_meta
    A[get_spinds(A)[I...]] = v
    A
end

@inline function add!(A::SpArray{T}, i::SpIndex, v::T) where {T}
    @boundscheck checkbounds(A, logicalindex(i))
    isactive(i) || return A
    @debug checkbounds(get_data(A), storageindex(i))
    @inbounds get_data(A)[storageindex(i)] += v
    A
end

@inline isactive(A::SpArray, I...) = (@_propagate_inbounds_meta; isactive(get_spinds(A), I...))

fillzero!(A::SpArray) = (fillzero!(A.data); A)

function update_sparsity!(A::SpArray, blkspy)
    A.shared_spinds && error("""
    The sparsity pattern is shared among some `SpArray`s. \
    Perhaps you should use `update_sparsity!(grid, blkspy)` instead of applying it to each `SpArray`.
    """)
    n = update_sparsity!(get_spinds(A), blkspy)
    resize_fillzero_data!(A, n)
    n
end

function resize_fillzero_data!(A::SpArray, n::Integer)
    resize_fillzero!(get_data(A), n)
    A
end
resize_fillzero_data!(A::AbstractMesh, n) = A

#############
# Broadcast #
#############

# Non-mutating broadcasts normally materialize a dense array over the logical
# domain.  We keep a sparse result only for simple zero-preserving operations
# whose flattened arguments are all SpArrays sharing the exact same SpIndices
# object (`===`), not merely an equal sparsity pattern.  Mutating broadcasts to
# a SpArray never change its sparsity; they write only into the destination's
# active storage.

Broadcast.BroadcastStyle(::Type{<: SpArray}) = ArrayStyle{SpArray}()

function Base.similar(bc::Broadcasted{ArrayStyle{SpArray}}, ::Type{ElType}) where {ElType}
    bc = Broadcast.instantiate(bc)
    bcf = Broadcast.flatten(bc)
    A = _first_sparray(bcf)
    _preserves_sparsity(bcf) ? similar_sparray(A, ElType) : similar(get_data(A), ElType, axes(bc))
end

similar_sparray(A::SpArray, ::Type{T}) where {T} = SpArray(similar(get_data(A), T, length(get_data(A))), get_spinds(A), true)

_first_sparray(A::SpArray) = A
_first_sparray(bc::Broadcasted) = _first_sparray(bc.args)
_first_sparray(::Tuple{}) = nothing
function _first_sparray(args::Tuple)
    A = _first_sparray(first(args))
    A === nothing ? _first_sparray(Base.tail(args)) : A
end
_first_sparray(x) = nothing

_all_sparrays(args::Tuple) = all(x -> x isa SpArray, args)
_preserves_sparsity(bc::Broadcasted) = _all_sparrays(bc.args) && identical_spinds(bc.args...) && _is_zero_preserving_bc_function(bc.f)
_is_zero_preserving_bc_function(f) = f in (+, -, *)

function Base.copyto!(dest::SpArray, bc::Broadcasted{ArrayStyle{SpArray}})
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    bc = Broadcast.instantiate(bc)
    bcf = Broadcast.flatten(bc)
    if identical_spinds(dest, bcf.args...)
        Base.copyto!(_get_data(dest), _get_data(bc))
    else
        _copyto_sp_broadcast!(get_device(dest), dest, bc)
    end
    dest
end

function _copyto_sp_broadcast!(::CPUDevice, dest::SpArray, bc::Broadcasted)
    @inbounds for i in activeindices(dest)
        dest[i] = bc[logicalindex(i)]
    end
    dest
end

@kernel function gpukernel_copyto_sp_broadcast!(dest, @Const(bc), @Const(spinds))
    k = @index(Global)
    active, i = _active_spindex(spinds, k)
    if active
        @inbounds dest[i] = bc[logicalindex(i)]
    end
end

function _copyto_sp_broadcast!(device::GPUDevice, dest::SpArray, bc::Broadcasted)
    backend = get_backend(device)
    spinds = get_spinds(dest)
    kernel = gpukernel_copyto_sp_broadcast!(backend)
    kernel(dest, bc, spinds; ndrange=_spindex_ndrange(spinds))
    dest
end

# Fast path for broadcasts over SpArrays with identical sparsity: unwrap the
# data vectors, but instantiate the broadcast here so GPU kernels do not infer
# broadcast axes from the sparse wrapper.
@inline _get_data(bc::Broadcasted{ArrayStyle{SpArray}}) = Broadcast.instantiate(Broadcast.broadcasted(bc.f, map(_get_data, bc.args)...))
@inline _get_data(x::SpArray) = get_data(x)
@inline _get_data(x::Any) = x

# helpers for copyto!
# all abstract arrays except SpArray and Tensor are not allowed in broadcasting
_ok(::Type{<: AbstractArray}) = false
_ok(::Type{<: SpArray}) = true
_ok(::Type{<: Tensor})  = true
_ok(::Type{<: Any})     = true
@generated function identical_spinds(args...)
    all(_ok, args) || return :(false)
    exps = [:(args[$i].spinds) for i in 1:length(args) if args[i] <: SpArray]
    n = length(exps)
    quote
        spindss = tuple($(exps...))
        @nall $n i -> spindss[1] === spindss[i]
    end
end

###############
# Custom show #
###############

struct CDot end
Base.show(io::IO, x::CDot) = print(io, "⋅")

struct ShowSpArray{T, N, A <: AbstractArray{T, N}, S} <: AbstractArray{T, N}
    parent::A
    summary_parent::S
end

# Array display scalar-indexes through `getindex`; show GPU-backed sparse arrays
# through a CPU copy while keeping the original object for the printed summary.
ShowSpArray(parent) = ShowSpArray(_show_parent(parent), parent)
_show_parent(parent) = get_device(parent) isa CPUDevice ? parent : cpu(parent)

Base.size(x::ShowSpArray) = size(x.parent)
Base.axes(x::ShowSpArray) = axes(x.parent)
@inline function Base.getindex(x::ShowSpArray, i::Integer...)
    @_propagate_inbounds_meta
    p = x.parent
    isactive(get_spinds(p)[i...]) ? maybecustomshow(p[i...]) : CDot()
end
maybecustomshow(x) = x
maybecustomshow(x::SpArray) = ShowSpArray(x)

Base.summary(io::IO, x::ShowSpArray) = summary(io, x.summary_parent)
Base.show(io::IO, mime::MIME"text/plain", x::SpArray) = show(io, mime, ShowSpArray(x))
Base.show(io::IO, x::SpArray) = show(io, ShowSpArray(x))
