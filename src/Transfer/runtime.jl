# -----------------------------------------------------------------------------
#  Transfer runtime
# -----------------------------------------------------------------------------

# `BasisWeightArray` is itself an `AbstractArray{<: BasisWeight}`, so its methods
# must be the more specific ones.
@inline function transfer_support_window(weights::BasisWeightArray, particles, p, mesh)
    @_propagate_inbounds_meta
    _transfer_support_window(Val(storageless(weights)), weights, particles, p, mesh)
end
@inline function transfer_support_window(weights::AbstractArray{<: BasisWeight}, particles, p, mesh)
    @_propagate_inbounds_meta
    supportnodes(weights[p])
end
@inline function _transfer_support_window(::Val{false}, weights, particles, p, mesh)
    @_propagate_inbounds_meta
    supportnodes_storage(weights)[p]
end
# Deferred weights store nothing, not even which nodes they cover, so the window
# is derived from the position the same way `update!` would have.
@inline function _transfer_support_window(::Val{true}, weights, particles, p, mesh)
    @_propagate_inbounds_meta
    supportnodes(basis(weights), LazyRow(particles, p), mesh)
end

# The per-step choice of `update!(...; deferred=true)` is applied once per
# transfer, on the host, before the launch: both arms are fully specialized, and
# for weights that cannot defer the second never exists.
@inline function select_weights(f::F, weights::W) where {F, W}
    if can_defer(W)
        isdeferred(weights) ? f(as_deferred(weights)) : f(weights)
    else
        f(weights)
    end
end

# `p` may be a linear index into a multi-dimensional particle space (FEM weights
# are quadrature point x cell), so the window storage supplies its shape.
@inline function weight_prop_view(weights::BasisWeightArray, ::Val{name}, p) where {name}
    @_propagate_inbounds_meta
    storage = supportnodes_storage(weights)
    A = getproperty(weights, name)
    view(A, nfill(:, Val(ndims(A) - ndims(storage)))..., particle_cartesian(storage, p))
end
@inline function weight_prop_view(weights::AbstractArray{<: BasisWeight}, ::Val{name}, p) where {name}
    @_propagate_inbounds_meta
    getproperty(weights[p], name)
end
@inline particle_cartesian(storage, p::Integer) = (@_propagate_inbounds_meta; CartesianIndices(storage)[p])
@inline particle_cartesian(storage, p::CartesianIndex) = p

# Deferred properties share one jet, evaluated to the highest order the equations
# reference rather than the order the weights were declared with, which keeps a
# transfer using only `w[ip]` cheap on `Order(2)` weights.
struct DeferredWeights{K, B, P, M, W, S, F}
    basis::B
    pt::P      # particle row: a basis may need more than the position
    mesh::M
    window::W
    state::S   # computed once per particle
    filter::F  # boundary filter, `nothing` for none
end
# The row is taken by value, not as a `LazyRow`: in a `@G2P2G` the support loop
# runs after the G2P half may have written `x[p]`, and a live row would then be
# read at the new position against the window taken at the old one.
DeferredWeights{K}(basis::B, pt::P, mesh::M, window::W, filter::F) where {K, B, P, M, W, F} =
    _deferred_weights(Order(K), basis, pt, mesh, window, filter)
@inline function _deferred_weights(::Order{K}, basis::B, pt::P, mesh::M, window::W, filter::F) where {K, B, P, M, W, F}
    @_propagate_inbounds_meta
    state = deferred_particle_state(Order(K), basis, pt, mesh, window, filter)
    DeferredWeights{K, B, P, M, W, typeof(state), F}(basis, pt, mesh, window, state, filter)
end

@inline deferred_particle_row(particles::StructArray, p) = (@_propagate_inbounds_meta; particles[p])
@inline deferred_particle_row(particles, p) = (@_propagate_inbounds_meta; LazyRow(particles, p))

@inline function deferred_jet(d::DeferredWeights{K}, ip) where {K}
    @_propagate_inbounds_meta
    deferred_node_jet(Order(K), d.basis, d.state, d.pt, d.mesh, d.window, d.filter, ip)
end

# Derivative order of a deferred property, `nothing` when it is stored, `missing`
# when absent. A plain function on types: a generated function may not call another.
function deferred_order(W::Type, name::Symbol)
    W <: BasisWeightArray || return nothing
    Vals = W.parameters[2]
    njets = W.parameters[6].parameters[1] + 1
    pos = findfirst(==(name), fieldnames(Vals))
    pos === nothing && return missing
    pos <= njets && fieldtype(Vals, pos) <: DeferredBasisValues ? pos - 1 : nothing
end

function split_weight_properties(W::Type, names)
    stored = Symbol[]; deferred = Symbol[]; orders = Int[]
    for n in names
        k = deferred_order(W, n)
        k === missing && return nothing
        k === nothing ? push!(stored, n) : (push!(deferred, n); push!(orders, k))
    end
    Tuple(stored), Tuple(deferred), orders
end

@generated function weight_columns(weights, ::Val{names}, particles, p, mesh, window) where {names}
    plan = split_weight_properties(weights, names)
    plan === nothing && return :(error("no such weight property among ", $(QuoteNode(names))))
    stored, _, orders = plan
    views = [:(weight_prop_view(weights, Val($(QuoteNode(n))), p)) for n in stored]
    deferred = isempty(orders) ? :nothing :
        :(DeferredWeights{$(maximum(orders))}(basis(weights), deferred_particle_row(particles, p), mesh, window, deferred_filter(weights)))
    quote
        @_propagate_inbounds_meta
        (stored = NamedTuple{$stored}(($(views...),)), deferred = $deferred)
    end
end

@generated function weight_node_values(weights, cols, ::Val{names}, ip) where {names}
    plan = split_weight_properties(weights, names)
    plan === nothing && return :(error("no such weight property among ", $(QuoteNode(names))))
    _, _, orders = plan
    exprs = map(names) do n
        k = deferred_order(weights, n)
        k === nothing ? :(cols.stored.$n[ip]) : :(jet[$(k + 1)])
    end
    jetexpr = isempty(orders) ? :() : :(jet = deferred_jet(cols.deferred, ip))
    quote
        @_propagate_inbounds_meta
        $jetexpr
        NamedTuple{$names}(($(exprs...),))
    end
end

# For an `SpGrid` the node index carries its storage slot as an `SpIndex`.
@inline function transfer_nodeindex(grid::SpGrid, window, ip)
    @_propagate_inbounds_meta
    spinds = get_spinds(grid)
    i = window[ip]
    @boundscheck checkbounds(spinds, i)
    @inbounds spinds[i]
end
@inline function transfer_nodeindex(grid, window, ip)
    @_propagate_inbounds_meta
    window[ip]
end

struct P2GSpGridStorageIndex
    i::Int
end

Base.@propagate_inbounds p2g_write_index(grid, i) = i
Base.@propagate_inbounds p2g_write_index(grid::Grid, i::CartesianIndex) = LinearIndices(grid)[i]
Base.@propagate_inbounds p2g_write_index(grid::SpGrid, i::CartesianIndex) = p2g_write_index(grid, get_spinds(grid)[Tuple(i)...])
# `@boundscheck` means debug mode only: the macros wrap the generated body in
# `@inbounds` outside `debug_mode`, and calling `update_sparsity!` first is the
# caller's obligation. Do not promote it to an unconditional check without
# re-measuring; making it fire in release costs a noticeable fraction of an
# `SpGrid` transfer, as does accumulating a flag to raise once per particle.
@inline function p2g_write_index(::SpGrid, i::SpIndex)
    si = storageindex(i)
    @boundscheck iszero(si) && error("@P2G: inactive SpGrid support node. Call update_sparsity! before @P2G.")
    P2GSpGridStorageIndex(si)
end

@inline function add!(A::SpArray{T}, i::P2GSpGridStorageIndex, v::T) where {T}
    @_propagate_inbounds_meta
    @debug checkbounds(get_data(A), i.i)
    @inbounds get_data(A)[i.i] += v
    A
end

@inline _atomic_index(A::HybridArray{<:Any, <:Any, <:SpArray}, i::P2GSpGridStorageIndex) = i.i

# Every equation in the grid-only part reads and writes only its own node, which
# is the same walk `@foreach` runs, so it shares the loops and kernels in foreach.jl.
P2G_nosum(f::F, device::AbstractDevice, schedule::Val, grid) where {F} =
    foreach_loop(f, device, schedule, grid)

function check_transfer_arguments(macroname, grid, particles, weights, partition)
    get_mesh(grid) isa AbstractMesh || error("$macroname: grid must have a mesh")
    eltype(weights) <: BasisWeight || error("$macroname: invalid `BasisWeight`s, got type $(typeof(weights))")
    if grid isa SpGrid
        eltype(weights) <: BasisWeight{CPDI} && cpdi_spgrid_error(macroname)
        if length(propertynames(grid)) > 1
            isempty(get_data(getproperty(grid, 2))) && error("$macroname: SpGrid indices not activated")
        end
    end
    @assert length(particles) ≤ length(weights)
    device = get_device(grid)
    @assert get_device(particles) == get_device(weights) == device
    check_partition_for_transfer(macroname, device, grid, weights, partition)
end

# A partition must live where the transfer runs: `BlockStrategy` schedules CPU
# threads, `GPUBlockStrategy` schedules GPU workgroups.
check_partition_for_transfer(macroname, ::CPUDevice, grid, weights, ::Nothing) = nothing
check_partition_for_transfer(macroname, ::GPUDevice, grid, weights, ::Nothing) = nothing
function check_partition_for_transfer(macroname, ::GPUDevice, grid, weights, partition::ThreadPartition{<: GPUBlockStrategy})
    macroname == "@P2G" || error("$macroname: the block-scheduled GPU transfer only supports @P2G so far. Use partitionless $macroname on GPU.")
    weights isa BasisWeightArray || error("$macroname: the block-scheduled GPU transfer requires weights from `generate_basis_weights`")
    check_partition_for_transfer(macroname, grid, weights, strategy(partition))
end
function check_partition_for_transfer(macroname, ::GPUDevice, grid, weights, partition)
    error("$macroname: this ThreadPartition lives on the CPU. Transfer it with `gpu(partition)` and `update!` it on the device.")
end
function check_partition_for_transfer(macroname, ::CPUDevice, grid, weights, ::ThreadPartition{<: GPUBlockStrategy})
    error("$macroname: this ThreadPartition lives on the GPU. Construct a CPU one with `ThreadPartition(mesh)`.")
end
function check_partition_for_transfer(macroname, ::CPUDevice, grid, weights, partition::ThreadPartition)
    check_partition_for_transfer(macroname, grid, weights, strategy(partition))
end
check_partition_for_transfer(macroname, grid, weights, strat) = nothing
function check_partition_for_transfer(macroname, grid, weights, strat::BlockStrategy)
    @assert nblocks(get_mesh(grid)) == nblocks(strat)
    if nassigned(strat) == 0
        error("$macroname: No particles assigned to any block in ThreadPartition")
    end
    check_partition_support(macroname, transfer_basis(weights), strat)
end
function check_partition_for_transfer(macroname, grid, weights, strat::GPUBlockStrategy)
    @assert nblocks(get_mesh(grid)) == nblocks(strat)
    if nactive(strat) == 0
        error("$macroname: No particles assigned to any block in ThreadPartition")
    end
    check_partition_support(macroname, basis(weights), strat)
end
function check_partition_support(macroname, b, strat)
    if support_width(b) > blockwidth(strat)
        error("$macroname: Block size for `ThreadPartition` is too small for basis $b. Increase `block_size_log2=Val(...)` on the `CartesianMesh` to ensure block size is ≥ kernel support.")
    end
end

# Reading `first(weights)` from a `BasisWeightArray` builds the row struct the
# file header calls out as expensive; the array itself already knows its basis.
transfer_basis(weights::AbstractArray) = basis(first(weights))
transfer_basis(weights::BasisWeightArray) = basis(weights)
