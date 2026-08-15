"""
    @foreach collection=>i begin
        statements...
    end

    @foreach collection[:,begin]=>i begin
        statements...
    end

Run `statements` for each index of `collection`.
Inside the block, `field[i]` is resolved to `collection.field[i]`, matching the
field-access convention used by transfer macros.
Use `\$(expr)` to evaluate an outer expression once before the generated loop.
Index the collection in the `collection=>i` argument to restrict the loop to a
slice, for example `grid[:,:,begin]=>i` or `grid[:,end]=>i`.

For `SpGrid`, only active sparse indices are visited. On GPU, the loop is
dispatched as a backend kernel.
"""
macro foreach(collection_i, body)
    foreach_expr(QuoteNode(:nothing), collection_i, body)
end

macro foreach(schedule::QuoteNode, collection_i, body)
    foreach_expr(schedule, collection_i, body)
end

function foreach_expr(schedule::QuoteNode, collection_i, body)
    collection, i, slice = parse_foreach_collection(collection_i)
    interpolations = Pair{Symbol, Any}[]
    body = extract_transfer_interpolations(body, interpolations)
    scope = TransferScope([collection=>i])
    body = resolve_refs(body, scope)
    if !DEBUG
        body = :(@inbounds $body)
    end
    code = if slice === nothing
        :(Tesserae.foreach_loop(($collection, $i) -> $body, Tesserae.get_device($collection), Val($schedule), $collection))
    else
        :(Tesserae.foreach_loop(($collection, $i) -> $body, Tesserae.get_device($collection), Val($schedule), $collection, $slice))
    end
    code = interpolate_transfer_values(code, TransferProgram(TransferEquation[], interpolations))
    esc(prettify(code; lines=true, alias=false))
end

function parse_foreach_collection(ex)
    if Meta.isexpr(ex, :call, 3) && ex.args[1] == :(=>)
        collection_expr, i = ex.args[2], ex.args[3]
        i isa Symbol || error("@foreach index must be a symbol, got `$i`")
        if Meta.isexpr(collection_expr, :ref)
            collection = first(collection_expr.args)
            collection isa Symbol || error("@foreach collection must be a symbol, got `$collection`")
            selectors = collection_expr.args[2:end]
            return collection, i, foreach_slice_expr(collection, selectors)
        else
            collection = collection_expr
            collection isa Symbol || error("@foreach collection must be a symbol, got `$collection`")
            return collection, i, nothing
        end
    else
        collection, i = unpair(ex)
        collection isa Symbol || error("@foreach collection must be a symbol, got `$collection`")
        i isa Symbol || error("@foreach index must be a symbol, got `$i`")
        return collection, i, nothing
    end
end

function foreach_slice_expr(collection, selectors)
    ranges = map(enumerate(selectors)) do (d, selector)
        if selector == :(:)
            :(Base.OneTo(size($collection, $d)))
        else
            selector = replace_foreach_slice_bounds(selector, collection, d)
            :(Tesserae.foreach_slice_range($selector))
        end
    end
    :(Tesserae.ForeachSlice(tuple($(ranges...))))
end

function replace_foreach_slice_bounds(expr, collection, d)
    MacroTools.postwalk(expr) do ex
        ex === :begin && return 1
        ex === :end && return :(size($collection, $d))
        ex
    end
end

struct ForeachSlice{N, R <: NTuple{N, AbstractRange{<: Integer}}}
    ranges::R
end

foreach_slice_range(index::Integer) = index:index
foreach_slice_range(range::AbstractRange{<: Integer}) = range
function foreach_slice_range(selector)
    throw(ArgumentError("@foreach slice indices must be `:`, integers, or integer ranges, got `$(selector)`"))
end

function foreach_check_slice(collection, slice::ForeachSlice)
    ndims(collection) == length(slice.ranges) ||
        throw(ArgumentError("@foreach slice has $(length(slice.ranges)) indices but collection has $(ndims(collection)) dimensions"))
    for d in eachindex(slice.ranges)
        foreach_check_slice_range(collection, slice.ranges[d], d)
    end
    nothing
end

function foreach_check_slice_range(collection, range, d)
    isempty(range) && return nothing
    bounds = Base.OneTo(size(collection, d))
    checkbounds(Bool, bounds, first(range)) || throw(BoundsError(collection, first(range)))
    checkbounds(Bool, bounds, last(range)) || throw(BoundsError(collection, last(range)))
    nothing
end

foreach_slice_ndrange(slice::ForeachSlice) = map(length, slice.ranges)

@inline function foreach_slice_index(slice::ForeachSlice, j::CartesianIndex)
    @inbounds CartesianIndex(map(getindex, slice.ranges, Tuple(j)))
end

@inline function foreach_slice_spindex(spinds, slice::ForeachSlice, j::CartesianIndex)
    I = foreach_slice_index(slice, j)
    @inbounds spinds[I]
end

foreach_indices(collection) = eachindex(collection)

# `@foreach` and the grid-only half of `@P2G` walk the same index spaces, and
# both run bodies that read and write only the node they are given, so they
# share the CPU loops below. `P2G_nosum` in transfer.jl is the `@P2G` entry
# point; the GPU kernels further down are shared the same way.
#
# Threading here does not hand single indices to `tforeach`. That leaves every
# worker iterating a `CartesianIndices` by linear index -- an integer division
# per dimension per node -- and gives up `@simd`. Splitting the index space
# instead, so that each worker iterates a sub-block of it, keeps both: 1.4x on a
# threaded 129^3 dense grid.

# `@foreach` sets no size threshold on threading: its body is arbitrary user
# code, so the item count says nothing about the work per item, and `@threaded`
# on it is an explicit request. `P2G_nosum` sets one, measured against the only
# bodies it ever runs.
function foreach_loop(f::F, ::CPUDevice, schedule::Val, collection) where {F}
    cpu_foreach_loop(f, schedule, collection, 0)
end

function foreach_loop(f::F, ::CPUDevice, schedule::Val, collection, slice::ForeachSlice) where {F}
    foreach_check_slice(collection, slice)
    cpu_foreach_slice_loop(f, schedule, collection, slice)
end

# `minthreaded` is the item count below which the fork-join costs more than the
# loop it replaces; zero threads whenever a scheduler is given.
function cpu_foreach_loop(f::F, schedule::Val, collection, minthreaded::Int) where {F}
    cpu_foreach(schedule, foreach_indices(collection), minthreaded) do i
        @inline f(collection, i)
    end
end

function cpu_foreach_loop(f::F, schedule::Val, collection::SpGrid, minthreaded::Int) where {F}
    cpu_foreach_blocks(schedule, get_spinds(collection), minthreaded) do i
        @inline f(collection, i)
    end
end

# A slice walks its own shape and maps each index back, so it shares the dense
# walk rather than the block walk even on an `SpGrid`.
function cpu_foreach_slice_loop(f::F, schedule::Val, collection, slice::ForeachSlice) where {F}
    cpu_foreach(schedule, CartesianIndices(foreach_slice_ndrange(slice)), 0) do j
        @inline f(collection, foreach_slice_index(slice, j))
    end
end

function cpu_foreach_slice_loop(f::F, schedule::Val, collection::SpGrid, slice::ForeachSlice) where {F}
    spinds = get_spinds(collection)
    cpu_foreach(schedule, CartesianIndices(foreach_slice_ndrange(slice)), 0) do j
        i = foreach_slice_spindex(spinds, slice, j)
        isactive(i) && @inline f(collection, i)
    end
end

# Dense index spaces: `eachindex` gives a `CartesianIndices` or a unit range,
# and a slice gives a `CartesianIndices` of its own shape.
function cpu_foreach(g::G, ::Val{scheduler}, inds, minthreaded::Int) where {G, scheduler}
    scheduler === :nothing && return foreach_subspace_loop(g, inds)
    d, nchunks = foreach_split(inds, Threads.nthreads())
    if nchunks < 2 || length(inds) < minthreaded
        return foreach_subspace_loop(g, inds)
    end
    n = size(inds, d)
    chunksize = cld(n, nchunks)
    tforeach(1:nchunks, scheduler) do chunk_id
        r = chunk_range(chunk_id, chunksize, n)
        isempty(r) || foreach_subspace_loop(g, foreach_subspace(inds, d, r))
    end
end

function foreach_subspace_loop(g::G, inds) where {G}
    @inbounds @simd for i in inds
        @inline g(i)
    end
end

# Split along the trailing axis, which keeps each chunk's nodes closest together
# in memory, and fall back to an earlier one only when it is too short to give
# every worker a piece: a slice can pin the trailing axis to a single index.
function foreach_split(inds, nworkers::Int)
    d = ndims(inds)
    d < 1 && return 1, 1
    while d > 1 && size(inds, d) < nworkers
        d -= 1
    end
    d, min(nworkers, size(inds, d))
end

# The `d`-th axis of `inds` restricted to positions `r`. Every axis is
# normalized to a `UnitRange` so that the result type does not depend on which
# axis a run happened to split.
@inline foreach_subspace(inds::AbstractUnitRange, ::Int, r::UnitRange{Int}) = inds[r]
@inline function foreach_subspace(inds::CartesianIndices{N}, d::Int, r::UnitRange{Int}) where {N}
    CartesianIndices(ntuple(Val(N)) do k
        ax = UnitRange(inds.indices[k])
        k == d ? ax[r] : ax
    end)
end

# An `SpGrid` walks blocks directly, in the order `ActiveSpIndices` yields them.
# Iterating that iterator instead carries its `(block, slot)` state through every
# node, and threading it used to mean collecting every active index into a
# `Vector{SpIndex}` first, since it is `SizeUnknown` and `tforeach` cannot index
# it. On a 129^3 grid with 11.6% of blocks active that collection is 26MB per
# call; walking the blocks is 5.3x sequential, 13.5x threaded, and allocates
# 5.7KB. Chunks are finer than the thread count so that clustered activity does
# not all land on one worker.
function cpu_foreach_blocks(g::G, ::Val{scheduler}, spinds::SpIndices, minthreaded::Int) where {G, scheduler}
    nblks = length(blocknumbering(spinds))
    nchunks = min(8 * Threads.nthreads(), nblks)
    if scheduler === :nothing || Threads.nthreads() == 1 || nchunks < 2 ||
       (minthreaded > 0 && active_node_count(spinds) < minthreaded)
        return foreach_blocks_loop(g, spinds, Base.OneTo(nblks))
    end
    chunksize = cld(nblks, nchunks)
    tforeach(1:nchunks, scheduler) do chunk_id
        foreach_blocks_loop(g, spinds, chunk_range(chunk_id, chunksize, nblks))
    end
end

function foreach_blocks_loop(g::G, spinds::SpIndices, blks) where {G}
    numbering = blocknumbering(spinds)
    blocks = CartesianIndices(numbering)
    localindices = CartesianIndices(blocksize(spinds))
    for b in blks
        @inbounds blocknumber = numbering[b]
        iszero(blocknumber) && continue
        @inbounds block = blocks[b]
        # `l` is the slot's linear position in the block: it indexes both
        # `localindices` and `SpArray.data`. `eachindex` would hand out
        # `CartesianIndex`es instead.
        for l in Base.OneTo(length(localindices))
            active, i = _active_spindex(spinds, blocknumber, block, l, localindices)
            active && @inline g(i)
        end
    end
end

# Slots owned by active blocks, which is what the block walk visits. No count is
# stored, so this is a pass over the block numbering: cheap next to the nodes it
# decides about, and skipped entirely when there is no threshold to test.
function active_node_count(spinds::SpIndices)
    numbering = blocknumbering(spinds)
    nactive = 0
    @inbounds for b in eachindex(numbering)
        nactive += !iszero(numbering[b])
    end
    nactive * blocklength(spinds)
end

@kernel function gpukernel_foreach(f, collection)
    i = @index(Global, Cartesian)
    @inline f(collection, i)
end

@kernel function gpukernel_foreach_linear(f, collection)
    i = @index(Global)
    @inline f(collection, i)
end

@kernel function gpukernel_foreach_spgrid(f, collection, @Const(spinds))
    k = @index(Global)
    active, i = _active_spindex(spinds, k)
    if active
        @inbounds @inline f(collection, i)
    end
end

@kernel function gpukernel_foreach_slice(f, collection, slice)
    j = @index(Global, Cartesian)
    i = foreach_slice_index(slice, j)
    @inline f(collection, i)
end

@kernel function gpukernel_foreach_slice_spgrid(f, collection, slice, @Const(spinds))
    j = @index(Global, Cartesian)
    i = foreach_slice_spindex(spinds, slice, j)
    if isactive(i)
        @inbounds @inline f(collection, i)
    end
end

function foreach_loop(f, device::GPUDevice, ::Val{scheduler}, collection) where {scheduler}
    scheduler == :nothing || @warn "Multi-threading is disabled for GPU" maxlog=1
    collection = collection isa QuadraturePoints ? parent(collection) : collection
    backend = get_backend(device)
    if collection isa SpGrid
        spinds = get_spinds(collection)
        kernel = gpukernel_foreach_spgrid(backend)
        kernel(f, collection, spinds; ndrange=_spindex_ndrange(spinds))
    elseif ndims(collection) == 1
        kernel = gpukernel_foreach_linear(backend)
        kernel(f, collection; ndrange=length(collection))
    else
        kernel = gpukernel_foreach(backend)
        kernel(f, collection; ndrange=size(collection))
    end
end

function foreach_loop(f, device::GPUDevice, ::Val{scheduler}, collection, slice::ForeachSlice) where {scheduler}
    scheduler == :nothing || @warn "Multi-threading is disabled for GPU" maxlog=1
    collection = collection isa QuadraturePoints ? parent(collection) : collection
    foreach_check_slice(collection, slice)
    ndrange = foreach_slice_ndrange(slice)
    backend = get_backend(device)
    if collection isa SpGrid
        spinds = get_spinds(collection)
        kernel = gpukernel_foreach_slice_spgrid(backend)
        kernel(f, collection, slice, spinds; ndrange)
    else
        kernel = gpukernel_foreach_slice(backend)
        kernel(f, collection, slice; ndrange)
    end
end
