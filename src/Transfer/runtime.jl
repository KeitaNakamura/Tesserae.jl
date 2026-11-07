# -----------------------------------------------------------------------------
#  Transfer runtime
# -----------------------------------------------------------------------------

struct StoredEvaluation end
struct DeferredEvaluation{F}
    filter::F
end

# A transfer carries resolved evaluation data, never the owner's mutable policy.
# The original value arrays also describe custom properties and conversion types;
# switching evaluation does not rebuild their storage or change their meaning.
struct TransferWeights{B, V <: NamedTuple, I, O <: Order, E}
    basis::B
    vals::V
    indices::I
    order::O
    evaluation::E
end
TransferWeights(weights::BasisWeightArray, evaluation) =
    TransferWeights(basis(weights), getfield(weights, :vals), supportnodes_storage(weights), derivative_order(weights), evaluation)

@inline basis(weights::TransferWeights) = weights.basis
@inline derivative_order(weights::TransferWeights) = weights.order
@inline supportnodes_storage(weights::TransferWeights) = weights.indices
@inline deferred_filter(weights::TransferWeights{<:Any,<:Any,<:Any,<:Any,<:DeferredEvaluation}) = weights.evaluation.filter

@inline function transfer_support_window(weights::TransferWeights, particles, p, mesh)
    @_propagate_inbounds_meta
    _transfer_support_window(weights.evaluation, weights, particles, p, mesh)
end
@inline _transfer_support_window(::StoredEvaluation, weights, particles, p, mesh) =
    (@_propagate_inbounds_meta; supportnodes_storage(weights)[p])
@inline _transfer_support_window(::DeferredEvaluation, weights, particles, p, mesh) =
    (@_propagate_inbounds_meta; supportnodes(basis(weights), LazyRows(particles)[p], mesh))

# `BasisWeightArray` is itself an `AbstractArray{<: BasisWeight}`, so its methods
# must be the more specific ones.
@inline function transfer_support_window(weights::BasisWeightArray, particles, p, mesh)
    @_propagate_inbounds_meta
    evaluation = storageless(weights) ? DeferredEvaluation(nothing) : StoredEvaluation()
    _transfer_support_window(evaluation, weights, particles, p, mesh)
end
@inline function transfer_support_window(weights::AbstractArray{<: BasisWeight}, particles, p, mesh)
    @_propagate_inbounds_meta
    supportnodes(weights[p])
end

# Construction and updates validate deferral support. The host selects the
# resolved mode once per transfer so both arms are fully specialized.
@inline function select_weights(f::F, weights::BasisWeightArray) where {F}
    if isdeferred(weights)
        f(TransferWeights(weights, DeferredEvaluation(deferred_filter(weights))))
    else
        f(TransferWeights(weights, StoredEvaluation()))
    end
end
@inline select_weights(f::F, weights::BasisWeightView) where {F} = select_weights(f, basis_weight_storage(weights))
@inline select_weights(f::F, weights) where {F} = f(weights)

# `p` may be a linear index into a multi-dimensional particle space (FEM weights
# are quadrature point x cell), so the window storage supplies its shape.
@inline function weight_prop_view(weights::Union{BasisWeightArray,TransferWeights}, ::Val{name}, p) where {name}
    @_propagate_inbounds_meta
    storage = supportnodes_storage(weights)
    A = getproperty(getfield(weights, :vals), name)
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
    W <: Union{BasisWeightArray,TransferWeights} || return nothing
    Vals = fieldtype(W, :vals)
    njets = _order_value(fieldtype(W, :order)) + 1
    pos = findfirst(==(name), fieldnames(Vals))
    pos === nothing && return missing
    deferred = W <: TransferWeights ? fieldtype(W, :evaluation) <: DeferredEvaluation :
                                     fieldtype(Vals, pos) <: DeferredBasisValues
    pos <= njets && deferred ? pos - 1 : nothing
end

@generated function weight_columns(weights, ::Val{names}, particles, p, mesh, window) where {names}
    orders = map(n -> deferred_order(weights, n), names)
    any(ismissing, orders) && return :(error("no such weight property among ", $(QuoteNode(names))))
    stored = Tuple(n for (n, k) in zip(names, orders) if k === nothing)
    order = maximum(filter(!isnothing, orders); init=-1)
    views = [:(weight_prop_view(weights, Val($(QuoteNode(n))), p)) for n in stored]
    deferred = order < 0 ? :nothing :
        :(DeferredWeights{$order}(basis(weights), deferred_particle_row(particles, p), mesh, window, deferred_filter(weights)))
    quote
        @_propagate_inbounds_meta
        (stored = NamedTuple{$stored}(($(views...),)), deferred = $deferred)
    end
end

@generated function weight_node_values(weights, cols, ::Val{names}, ip) where {names}
    orders = map(n -> deferred_order(weights, n), names)
    any(ismissing, orders) && return :(error("no such weight property among ", $(QuoteNode(names))))
    exprs = map(names, orders) do n, k
        if k === nothing
            :(cols.stored.$n[ip])
        else
            T = eltype(fieldtype(fieldtype(weights, :vals), n))
            :(convert($T, jet[$(k + 1)]))
        end
    end
    jetexpr = all(isnothing, orders) ? :() : :(jet = deferred_jet(cols.deferred, ip))
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
    length(particles) ≤ length(weights) ||
        error("$macroname: `weights` must cover the particles: got $(length(weights)) weights for $(length(particles)) particles")
    device = get_device(grid)
    get_device(particles) == get_device(weights) == device ||
        error("$macroname: grid, particles, and weights must be on the same device: got $device, $(get_device(particles)), $(get_device(weights))")
    check_partition_for_transfer(macroname, device, grid, particles, weights, partition)
end

# A partition must live where the transfer runs: `CPUBlockStrategy` schedules CPU
# threads, `GPUBlockStrategy` schedules GPU workgroups.
check_partition_for_transfer(macroname, ::CPUDevice, grid, particles, weights, ::Nothing) = nothing
check_partition_for_transfer(macroname, ::GPUDevice, grid, particles, weights, ::Nothing) = nothing
function check_partition_for_transfer(macroname, ::GPUDevice, grid, particles, weights, partition::Partition{<: GPUBlockStrategy})
    macroname == "@P2G" || error("$macroname: the block-scheduled GPU transfer only supports @P2G so far. Use partitionless $macroname on GPU.")
    weights isa Union{BasisWeightArray,BasisWeightView} || error("$macroname: the block-scheduled GPU transfer requires weights from `generate_basis_weights`")
    check_partition_for_transfer(macroname, grid, particles, weights, strategy(partition))
end
function check_partition_for_transfer(macroname, ::GPUDevice, grid, particles, weights, partition)
    error("$macroname: this Partition lives on the CPU. Transfer it with `gpu(partition)` and `update!` it on the device.")
end
function check_partition_for_transfer(macroname, ::CPUDevice, grid, particles, weights, ::Partition{<: GPUBlockStrategy})
    error("$macroname: this Partition lives on the GPU. Construct a CPU one with `Partition(mesh)`.")
end
function check_partition_for_transfer(macroname, ::CPUDevice, grid, particles, weights, partition::Partition)
    check_partition_for_transfer(macroname, grid, particles, weights, strategy(partition))
end
check_partition_for_transfer(macroname, grid, particles, weights, strat) = nothing
function check_partition_for_transfer(macroname, grid, particles, weights, strat::BlockStrategy)
    nblocks(get_mesh(grid)) == nblocks(strat) ||
        error("$macroname: the grid's mesh does not match the mesh the `Partition` was built for")
    if nassigned(strat) == 0
        error("$macroname: No particles assigned to any block in Partition")
    end
    length(strat.particleindices) == length(particles) ||
        error("$macroname: `update!(partition, particles.x)` must run with these particles before the transfer")
    check_partition_support(macroname, transfer_basis(weights), strat)
end
function check_partition_support(macroname, b, strat)
    if support_width(b) > blockwidth(strat)
        error("$macroname: Block size for `Partition` is too small for basis $b. Increase `block_size_log2=Val(...)` on the `CartesianMesh` to ensure block size is ≥ kernel support.")
    end
end

# Reading `first(weights)` from a `BasisWeightArray` builds the row struct the
# weight-references note in program.jl calls out as expensive; the array itself
# already knows its basis.
transfer_basis(weights::AbstractArray) = basis(first(weights))
transfer_basis(weights::BasisWeightArray) = basis(weights)

transfer_basis(weights::BasisWeightView) = basis(weights)
