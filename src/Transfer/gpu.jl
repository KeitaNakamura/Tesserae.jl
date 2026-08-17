# -----------------------------------------------------------------------------
#  GPU transfer kernels
# -----------------------------------------------------------------------------

# ---- shared-memory tiles ----

# A support window starts at most one node before the particle's cell, so the tile
# reaches one node past the block even at width 1: `BSpline(Constant())` on a
# particle in the last cell picks the next block's first node.
@inline p2g_tile_halo(support_width::Integer) = max(support_width - 1, 1)

function p2g_tile_contains(basis, mesh::CartesianMesh{dim}, window::CartesianIndices{dim}, block::CartesianIndex{dim}) where {dim}
    bw = blockwidth(mesh)
    halo = p2g_tile_halo(support_width(basis))
    all(ntuple(Val(dim)) do d
        lo = (block[d] - 1) * bw - halo + 1
        lo <= first(window.indices[d]) && last(window.indices[d]) <= lo + bw + 2*halo - 1
    end)
end

# The two lowerings of one `@P2G` scatter: `particle` writes straight to the grid,
# `tile` accumulates into the block-scheduled GPU kernel's shared-memory tile.
struct P2GBodies{names, F, FT}
    particle::F
    tile::FT
end
P2GBodies(particle::F, tile::FT, ::Val{names}) where {names, F, FT} = P2GBodies{names, F, FT}(particle, tile)
scattered_names(::P2GBodies{names}) where {names} = names

# Matches the layout `HybridArray`'s flatten/add! use.
@inline tile_components(v::Number) = (v,)
@inline tile_components(v::Union{Tensor, StaticArray}) = Tuple(v)
# Structural, so the generators below can call it: a generated function may not
# call another, which rules out anything touching `zero(T)` for Tensorial types.
@inline tile_ncomps(::Type{T}) where {T <: Number} = 1
@inline tile_ncomps(::Type{Tensor{S, T, N, L}}) where {S, T, N, L} = L
@inline tile_ncomps(::Type{SV}) where {SV <: StaticArray} = prod(size(SV))

# A tile stores `SIDE^dim` nodes per scalar component, component-major by field.
# `origin` sits one node before the tile's first node in every axis.
@inline function tile_slot(node::CartesianIndex{dim}, origin::CartesianIndex{dim}, ::Val{SIDE}) where {dim, SIDE}
    slot = node[dim] - origin[dim]
    for d in dim-1:-1:1
        slot = (slot - 1) * SIDE + (node[d] - origin[d])
    end
    slot
end
@inline function tile_node(k::Integer, origin::CartesianIndex{dim}, ::Val{SIDE}) where {dim, SIDE}
    local_index = CartesianIndices(nfill(SIDE, Val(dim)))[k]
    CartesianIndex(ntuple(d -> origin[d] + local_index[d], Val(dim)))
end

@inline function tile_add!(tile, ::Val{TILELEN}, fieldoffset::Int, slot::Integer, v) where {TILELEN}
    data = tile_components(v)
    for j in eachindex(data)
        @inbounds Atomix.@atomic tile[(fieldoffset + j - 1) * TILELEN + slot] += data[j]
    end
end

# The scatter and the merge must agree on where each field starts, so it is
# derived once here. Structural, so the generators below can call it.
function tile_field_offsets(::Type{G}, names::Tuple) where {G}
    offsets = Int[]
    offset = 0
    for name in names
        push!(offsets, offset)
        offset += tile_ncomps(fieldtype(eltype(G), name))
    end
    Tuple(offsets)
end
# The macro-emitted body has no types to lay the tile out with.
@generated tile_offsets(grid, ::Val{names}) where {names} =
    Expr(:tuple, tile_field_offsets(grid, names)...)

@inline tile_rebuild(::Type{T}, comps) where {T <: Number} = comps[1]
@inline tile_rebuild(::Type{T}, comps) where {T <: Union{Tensor, StaticArray}} = T(comps...)

# Fully unrolled: the GPU compiler must see every field eltype as a constant, and
# a generated body may not contain closures, which rules out `ntuple` with a lambda.
@generated function merge_tile_node!(grid, tile, ::Val{names}, ::Val{TILELEN}, slot, node) where {names, TILELEN}
    exprs = Any[]
    for (name, offset) in zip(names, tile_field_offsets(grid, names))
        T = fieldtype(eltype(grid), name)
        elems = [:(tile[$(offset + j - 1) * TILELEN + slot]) for j in 1:tile_ncomps(T)]
        push!(exprs, quote
            comps = @inbounds ($(elems...),)
            if !all(iszero, comps)
                @inbounds add!(grid.$name, p2g_write_index(grid, node), tile_rebuild($T, comps))
            end
        end)
    end
    quote
        @_inline_meta
        $(exprs...)
        nothing
    end
end

# One tile holds every scattered field, so their scalar types must agree.
@generated tile_total_comps(grid, ::Val{names}) where {names} =
    sum(name -> tile_ncomps(fieldtype(eltype(grid), name)), names)
@generated function tile_scalartype(grid, ::Val{names}) where {names}
    types = map(name -> eltype(fieldtype(eltype(grid), name)), names)
    allequal(types) || return :(error("@P2G: block-scheduled transfer requires one scalar type across the scattered fields, got ", $(join(unique(types), ", "))))
    first(types)
end

# ---- kernels and dispatch ----

# `@P2G`, `@G2P` and `@G2P2G` all walk the same four arguments, so one kernel
# serves all three, each call site compiling its own copy of `f`.
#
# `@Const` is deliberately absent from the container arguments: KernelAbstractions
# rewrites the argument's type, and indexing a `StructArray`, `BasisWeightArray`
# or mesh whose arrays became `Const` wrappers costs far more than the read-only
# cache hint wins. Unpacking the arrays to annotate each one gains nothing either,
# a particle's weights being read by that particle alone and the grid reads being
# absorbed by L2 already.
@kernel function gpukernel_transfer(f, grid, particles, weights)
    p = @index(Global)
    @inline f(grid, particles, weights, p)
end
function P2G(f, device::GPUDevice, ::Val{scheduler}, grid, particles, weights, ::Nothing, zeroed::Tuple=()) where {scheduler}
    scheduler == :nothing || @warn "Multi-threading is disabled for GPU" maxlog=1
    fillzero_each!(device, zeroed)
    particles = particles isa QuadraturePoints ? parent(particles) : particles
    backend = get_backend(device)
    kernel = gpukernel_transfer(backend)
    kernel(f, hybrid(grid, device), particles, weights; ndrange=length(particles))
end

# GPU with a device partition: one workgroup per nonempty grid block, merging a
# shared tile into the grid with a handful of global atomics per node instead of
# one per particle-node pair. Accumulation order within a block follows the
# partition, so the result is not bitwise reproducible between runs, exactly like
# the atomic particle-parallel path.
const P2G_BLOCK_GROUPSIZE = 128

@kernel function gpukernel_P2G_blocks(tilebody, grid, particles, weights,
                                      @Const(particleindices), @Const(offsets), @Const(blocklist),
                                      ::Type{Tt}, ::Val{names}, ::Val{SIDE}, ::Val{TILELEN}, ::Val{TOTAL},
                                      ::Val{BW}, ::Val{HALO}, blkdims::Dims{dim}) where {Tt, names, SIDE, TILELEN, TOTAL, BW, HALO, dim}
    grp = @index(Group, Linear)
    l = Int(@index(Local, Linear))
    tile = @localmem Tt (TOTAL,)
    k = l
    while k <= TOTAL
        @inbounds tile[k] = zero(Tt)
        k += P2G_BLOCK_GROUPSIZE
    end
    @synchronize
    @inbounds b = Int(blocklist[grp])
    blockcoord = CartesianIndices(blkdims)[b]
    origin = CartesianIndex(ntuple(d -> (blockcoord[d] - 1) * BW - HALO, Val(dim)))
    @inbounds pstart = Int(offsets[b])
    @inbounds pstop = Int(offsets[b+1])
    k = l
    while k <= pstop - pstart
        @inbounds p = Int(particleindices[pstart + k])
        @inline tilebody(tile, origin, Val(SIDE), Val(TILELEN), grid, particles, weights, p)
        k += P2G_BLOCK_GROUPSIZE
    end
    @synchronize
    gridsize = size(grid)
    k = l
    while k <= TILELEN
        node = tile_node(k, origin, Val(SIDE))
        if all(ntuple(d -> 1 <= node[d] <= gridsize[d], Val(dim)))
            @inline merge_tile_node!(grid, tile, Val(names), Val(TILELEN), k, node)
        end
        k += P2G_BLOCK_GROUPSIZE
    end
end

function P2G(bodies::P2GBodies, device::GPUDevice, ::Val{scheduler}, grid, particles, weights, partition::ThreadPartition{<: GPUBlockStrategy}, zeroed::Tuple=()) where {scheduler}
    scheduler == :nothing || @warn "Multi-threading is disabled for GPU" maxlog=1
    fillzero_each!(device, zeroed)
    particles = particles isa QuadraturePoints ? parent(particles) : particles
    bs = strategy(partition)
    names = Val(scattered_names(bodies))
    length(bs.particleindices) == length(particles) ||
        error("@P2G: `update!(partition, particles.x)` must run with these particles before the transfer")
    backend = get_backend(device)
    Tt = tile_scalartype(grid, names)
    sw = support_width(basis(weights))
    BW = blockwidth(bs)
    dim = length(nblocks(bs))
    halo = p2g_tile_halo(sw)
    side = BW + 2*halo
    tilelen = side^dim
    total = tilelen * tile_total_comps(grid, names)
    if total * sizeof(Tt) > 32768
        @warn "@P2G: shared-memory tile ($(total * sizeof(Tt)) B) exceeds the block-scheduled budget; falling back to the particle-parallel path" maxlog=1
        return P2G(bodies.particle, device, Val(scheduler), grid, particles, weights, nothing)
    end
    kernel = gpukernel_P2G_blocks(backend, (P2G_BLOCK_GROUPSIZE,))
    kernel(bodies.tile, hybrid(grid, device), particles, weights,
           bs.particleindices, bs.offsets, bs.blocklist,
           Tt, names, Val(side), Val(tilelen), Val(total), Val(BW), Val(halo), nblocks(bs);
           ndrange=P2G_BLOCK_GROUPSIZE * nactive(bs))
end

# The grid goes in unwrapped: nothing in a `@G2P` body scatters, so it needs no atomics.
function G2P(f, device::GPUDevice, ::Val{scheduler}, grid, particles, weights) where {scheduler}
    scheduler == :nothing || @warn "Multi-threading is disabled for GPU" maxlog=1
    particles = particles isa QuadraturePoints ? parent(particles) : particles
    backend = get_backend(device)
    kernel = gpukernel_transfer(backend)
    kernel(f, grid, particles, weights; ndrange=length(particles))
end
