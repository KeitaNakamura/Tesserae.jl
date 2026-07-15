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
foreach_indices(collection::SpGrid) = activeindices(get_spinds(collection))

function foreach_loop(f, ::CPUDevice, ::Val{scheduler}, collection) where {scheduler}
    tforeach(foreach_indices(collection), scheduler) do i
        @inline f(collection, i)
    end
end

function foreach_loop(f, ::CPUDevice, ::Val{:nothing}, collection::SpGrid)
    for i in foreach_indices(collection)
        @inline f(collection, i)
    end
end

function foreach_loop(f, ::CPUDevice, ::Val{scheduler}, collection::SpGrid) where {scheduler}
    tforeach(collect(foreach_indices(collection)), scheduler) do i
        @inline f(collection, i)
    end
end

function foreach_loop(f, ::CPUDevice, ::Val{scheduler}, collection, slice::ForeachSlice) where {scheduler}
    foreach_check_slice(collection, slice)
    ndrange = foreach_slice_ndrange(slice)
    tforeach(CartesianIndices(ndrange), scheduler) do j
        i = foreach_slice_index(slice, j)
        @inline f(collection, i)
    end
end

function foreach_loop(f, ::CPUDevice, ::Val{scheduler}, collection::SpGrid, slice::ForeachSlice) where {scheduler}
    foreach_check_slice(collection, slice)
    ndrange = foreach_slice_ndrange(slice)
    spinds = get_spinds(collection)
    tforeach(CartesianIndices(ndrange), scheduler) do j
        i = foreach_slice_spindex(spinds, slice, j)
        isactive(i) && @inline f(collection, i)
    end
end

@kernel function gpukernel_foreach(f, collection)
    i = @index(Global, Cartesian)
    f(collection, i)
end

@kernel function gpukernel_foreach_linear(f, collection)
    i = @index(Global)
    f(collection, i)
end

@kernel function gpukernel_foreach_spgrid(f, collection, @Const(spinds))
    k = @index(Global)
    active, i = _active_spindex(spinds, k)
    if active
        @inbounds f(collection, i)
    end
end

@kernel function gpukernel_foreach_slice(f, collection, slice)
    j = @index(Global, Cartesian)
    i = foreach_slice_index(slice, j)
    f(collection, i)
end

@kernel function gpukernel_foreach_slice_spgrid(f, collection, slice, @Const(spinds))
    j = @index(Global, Cartesian)
    i = foreach_slice_spindex(spinds, slice, j)
    if isactive(i)
        @inbounds f(collection, i)
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
