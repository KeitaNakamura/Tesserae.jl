"""
    @foreach collection=>i begin
        statements...
    end

Run `statements` for each index of `collection`.
Inside the block, `field[i]` is resolved to `collection.field[i]`, matching the
field-access convention used by transfer macros.
Use `\$(expr)` to evaluate an outer expression once before the generated loop.

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
    collection, i = unpair(collection_i)
    interpolations = Pair{Symbol, Any}[]
    body = extract_transfer_interpolations(body, interpolations)
    scope = TransferScope([collection=>i])
    body = resolve_refs(body, scope)
    if !DEBUG
        body = :(@inbounds $body)
    end
    code = quote
        Tesserae.foreach_loop(($collection, $i) -> $body,
                              Tesserae.get_device($collection),
                              Val($schedule), $collection)
    end
    code = interpolate_transfer_values(code, TransferProgram(TransferEquation[], interpolations))
    esc(prettify(code; lines=true, alias=false))
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

function foreach_loop(f, device::GPUDevice, ::Val{scheduler}, collection) where {scheduler}
    scheduler == :nothing || @warn "Multi-threading is disabled for GPU" maxlog=1
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
    synchronize(backend)
end
