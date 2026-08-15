using MacroTools

struct TransferEquation
    kind::Symbol
    lhs::Any
    rhs::Any
    op::Symbol
end

is_sum(eq::TransferEquation) = eq.kind === :sum

struct TransferProgram
    equations::Vector{TransferEquation}
    interpolations::Vector{Pair{Symbol, Any}}
end

function split_sum_equations(program::TransferProgram, macroname::String)
    equations = program.equations
    issum = map(is_sum, equations)
    if !allequal(issum) && !issorted(issum; rev=true)
        error("$macroname: Equations without `@∑` must come after those with `@∑`")
    end
    equations[issum], equations[.!issum]
end

# A weight reference like `w[ip]` resolves against the parent weights array:
# through a per-particle column view bound once outside the support loop when
# the scope caches replacements, as `weights.w[ip, p]` inline otherwise. The
# per-particle `BasisWeight` this used to build kept enough live SubArray state
# across the loop to spill GPU registers into local memory, and that spill
# traffic -- not the interpolation -- dominated the transfer kernels.
struct TrailingIndexed
    parent::Any
    trailing::Any
end

struct TransferScope
    bindings::Dict{Any,Any}
    replacements::Union{Nothing,Dict{Any,Vector{Expr}}}
end

function TransferScope(maps::Vector{<: Pair}; cache::Bool=false)
    bindings = Dict{Any,Any}()
    for map in maps
        parent, index = map
        haskey(bindings, index) && error("transfer index `$index` is bound more than once")
        bindings[index] = parent
    end
    replacements = cache ? Dict{Any,Vector{Expr}}(index => Expr[] for index in keys(bindings)) : nothing
    TransferScope(bindings, replacements)
end

uncached(scope::TransferScope) = TransferScope(scope.bindings, nothing)

function cached_replacements(scope::TransferScope, indices...)
    scope.replacements === nothing && error("reference cache is not enabled for this transfer scope")
    exprs = Expr[]
    for index in indices
        haskey(scope.replacements, index) || error("index `$index` is not bound in this transfer scope")
        union!(exprs, scope.replacements[index])
    end
    exprs
end

function resolve_equation(eq::TransferEquation, scope::TransferScope)
    TransferEquation(eq.kind, resolve_refs(eq.lhs, scope), resolve_refs(eq.rhs, scope), eq.op)
end

function resolve_sum_equations(equations::Vector{TransferEquation}, scope::TransferScope, macroname::String, index)
    lhs_scope = uncached(scope)
    map(equations) do eq
        @capture(eq.lhs, name_Symbol[idx_]) || error("$macroname: invalid LHS in `@∑` equation: $(eq.lhs)")
        idx == index || error("$macroname: invalid LHS index in `@∑` equation: $(eq.lhs) (must be [$index])")
        TransferEquation(eq.kind, resolve_refs(eq.lhs, lhs_scope), resolve_refs(eq.rhs, scope), eq.op)
    end
end

function push_unique!(xs::Vector, x)
    x in xs || push!(xs, x)
    xs
end

# `@G2P2G` fuses a G2P and a P2G loop into a single pass over the particles, so
# the support window must be bound once and shared by both halves. The symbol
# travels with a `load` flag: whichever half holds `load=true` emits the
# binding statement, the other just uses it.
struct SupportWindowBinding
    window::Symbol
    load::Bool
end
SupportWindowBinding() = SupportWindowBinding(gensym(:window), true)
SupportWindowBinding(binding::SupportWindowBinding; load::Bool) = SupportWindowBinding(binding.window, load)

function support_window_exprs(binding::SupportWindowBinding, weights, p)
    binding.load || return ()
    (:($(binding.window) = Tesserae.transfer_support_window($weights, $p)),)
end

# The per-particle support window, straight from the weights storage. A plain
# array of `BasisWeight`s -- the macros accept any container whose `weights[p]`
# is a `BasisWeight` -- goes through the element instead. `BasisWeightArray` is
# itself such an array, so its methods must be the more specific ones.
@inline function transfer_support_window(weights::BasisWeightArray, p)
    @_propagate_inbounds_meta
    supportnodes_storage(weights)[p]
end
@inline function transfer_support_window(weights::AbstractArray{<: BasisWeight}, p)
    @_propagate_inbounds_meta
    supportnodes(weights[p])
end

# One weight property's column for particle `p`: the whole leading (support)
# block, with the particle indices fixed. `p` may be a linear index into a
# multi-dimensional particle space (FEM weights are quadrature point x cell),
# so the support-window storage supplies the particle dimensionality.
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

# One support node of the window, resolved against the grid: an `SpIndex`
# carrying the storage slot for an `SpGrid`, the plain index otherwise.
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

# Grid properties referenced with the node index, straight from the equation
# syntax. The kernels are handed a grid narrowed to these, because every extra
# `SpArray` field in the argument -- referenced or not -- measurably slows a
# GPU transfer kernel down.
function collect_transfer_refs(equations, index)
    names = Symbol[]
    for eq in equations
        for expr in (eq.lhs, eq.rhs)
            MacroTools.postwalk(expr) do ex
                @capture(ex, x_Symbol[i_]) && i == index && push_unique!(names, x)
                ex
            end
        end
    end
    names
end

narrowed_grid_expr(grid, equations, index) =
    :(Tesserae.narrow_transfer_grid($grid, Val($(Expr(:tuple, map(QuoteNode, collect_transfer_refs(equations, index))...)))))

# Keep the mesh (always the first property) so the result stays a grid, and at
# least one array component so an `SpGrid` stays an `SpGrid` for dispatch and
# `get_spinds`. Non-StructArray grids pass through untouched.
narrow_transfer_grid(grid, ::Val) = grid
@generated function narrow_transfer_grid(grid::StructArray{<: Any, <: Any, <: NamedTuple{names}}, ::Val{refs}) where {names, refs}
    keep = Symbol[first(names)]
    for name in Base.tail(names)
        name in refs && push!(keep, name)
    end
    length(keep) == 1 && length(names) > 1 && push!(keep, names[2])
    Tuple(keep) == names && return :grid
    fields = [:(getproperty(grid, $(QuoteNode(name)))) for name in keep]
    :(StructArray(NamedTuple{$(Tuple(keep))}(($(fields...),))))
end

# All four transfer macros take the same shape: an optional `:schedule`
# QuoteNode, three `parent => index` pairs, an optional partition, and the
# equation block. Parsing it once keeps them from drifting, and lets a wrong
# call say what the right one looks like instead of listing `::Any` candidates.
function parse_transfer_macro_args(macroname, args, allow_partition::Bool)
    args = collect(args)
    schedule = QuoteNode(:nothing)
    if !isempty(args) && first(args) isa QuoteNode
        schedule = popfirst!(args)
    end
    partition = nothing
    if length(args) == 5
        allow_partition || throw(ArgumentError(transfer_macro_usage(macroname, allow_partition)))
        partition = args[4]
    elseif length(args) != 4
        throw(ArgumentError(transfer_macro_usage(macroname, allow_partition)))
    end
    schedule, args[1], args[2], args[3], partition, last(args)
end

function transfer_macro_usage(macroname, allow_partition)
    indices = macroname == "@P2G_Matrix" ? "grid=>(i,j) particles=>p weights=>(ip,jp)" :
                                           "grid=>i particles=>p weights=>ip"
    part = allow_partition ? " [partition]" : ""
    "$macroname: expected `$macroname [:schedule] $indices$part begin ... end`"
end

"""
    @P2G grid=>i particles=>p weights=>ip [partition] begin
        equations...
    end

Particle-to-grid transfer macro.
Based on the `parent => index` expressions, `a[index]` in `equations`
translates to `parent.a[index]`. This `index` can be replaced with
any other name.

# Examples
```julia
@P2G grid=>i particles=>p weights=>ip begin

    # Particle-to-grid transfer
    m[i]  = @∑ w[ip] * m[p]
    mv[i] = @∑ w[ip] * m[p] * v[p]
    f[i]  = @∑ -V[p] * σ[p] * ∇w[ip]

    # Calculation on grid
    vⁿ[i] = mv[i] / m[i]
    v[i]  = vⁿ[i] + (f[i] / m[i]) * Δt

end
```

This expands to roughly the following code:

```julia
# Reset grid properties
@. grid.m  = zero(grid.m)
@. grid.mv = zero(grid.mv)
@. grid.f  = zero(grid.f)

# Particle-to-grid transfer
for p in eachindex(particles)
    bw = weights[p]
    nodeindices = supportnodes(bw)
    for ip in eachindex(nodeindices)
        i = nodeindices[ip]
        grid.m [i] += bw.w[ip] * particles.m[p]
        grid.mv[i] += bw.w[ip] * particles.m[p] * particles.v[p]
        grid.mv[i] += -particles.V[p] * particles.σ[p] * bw.∇w[ip]
    end
end

# Calculation on grid
for i in eachindex(grid)
    grid.vⁿ[i] = grid.mv[i] / grid.m[i]
    grid.v[i]  = grid.vⁿ[i] + (grid.f[i] / grid.m[i]) * Δt
end
```

Use `\$(expr)` inside transfer equations to evaluate an outer expression once
before the generated transfer loops and use the captured value in the loop body.
For example, `\$Δt` captures the current value of `Δt`.

!!! warning
    In `@P2G`, `Calculation on grid` part must be placed after
    `Particle-to-grid transfer` part.
"""
macro P2G(args...)
    P2G_expr(parse_transfer_macro_args("@P2G", args, true)...)
end

function P2G_expr(schedule::QuoteNode, grid_i::Expr, particles_p::Expr, weights_ip::Expr, partition, equations::Expr)
    P2G_expr(schedule, unpair(grid_i), unpair(particles_p), unpair(weights_ip), partition, parse_transfer_program(equations))
end

function P2G_expr(schedule::QuoteNode, (grid,i), (particles,p), (weights,ip), partition, program::TransferProgram)
    sum_equations, nosum_equations = split_sum_equations(program, "@P2G")

    code = quote
        Tesserae.check_transfer_arguments("@P2G", $grid, $particles, $weights, $partition)
    end

    if !isempty(sum_equations)
        zeroed, body = P2G_sum_expr(schedule, (grid,i), (particles,p), (weights,ip), sum_equations)
        if !DEBUG
            body = :(@inbounds $body)
        end
        code = quote
            $code
            Tesserae.P2G(($grid, $particles, $weights, $p) -> $body, Tesserae.get_device($grid), Val($schedule), $(narrowed_grid_expr(grid, sum_equations, i)), $particles, $weights, $partition, $zeroed)
        end
    end

    if !isempty(nosum_equations)
        body = P2G_nosum_expr((grid,i), nosum_equations)
        if !DEBUG
            body = :(@inbounds $body)
        end
        code = quote
            $code
            Tesserae.P2G_nosum(($grid, $i) -> $body, Tesserae.get_device($grid), Val($schedule), $(narrowed_grid_expr(grid, nosum_equations, i)))
        end
    end

    code = interpolate_transfer_values(code, program)
    esc(prettify(code; lines=true, alias=false))
end

function P2G_sum_expr(schedule::QuoteNode, (grid,i), (particles,p), (weights,ip), sum_equations::Vector, binding::SupportWindowBinding=SupportWindowBinding())
    @gensym gridwriteindex
    (; window) = binding

    scope = TransferScope([grid=>i, particles=>p, TrailingIndexed(weights, p)=>ip]; cache=true)
    sum_equations = resolve_sum_equations(sum_equations, scope, "@P2G", i)
    particle_replacements = cached_replacements(scope, p)
    inner_replacements = cached_replacements(scope, i, ip)
    inner_symbols = p2g_cached_symbols(inner_replacements)

    fillzero_targets = Any[]
    hoist_exprs = Any[]
    sum_exprs = Any[]
    for eq in sum_equations
        (; lhs, rhs, op) = eq
        op == :(=)  && push_unique!(fillzero_targets, remove_indexing(lhs))
        op == :(-=) && (rhs = :(-$rhs))
        rhs = hoist_p2g_rhs!(hoist_exprs, inner_symbols, rhs)
        push!(sum_exprs, p2g_sum_add_expr(lhs, gridwriteindex, rhs))
    end

    body = quote
        $(particle_replacements...)
        $(hoist_exprs...)
        $(support_window_exprs(binding, weights, p)...)
        for $ip in eachindex($window)
            $i = Tesserae.transfer_nodeindex($grid, $window, $ip)
            $gridwriteindex = Tesserae.p2g_write_index($grid, $i)
            $(inner_replacements...)
            $(sum_exprs...)
        end
    end

    # All the assigned fields as one tuple, handed to `P2G` rather than zeroed
    # ahead of it: that lets the threaded path zero them inside the parallel
    # region it already opens, instead of paying a fork-join of its own.
    Expr(:tuple, fillzero_targets...), body
end

function p2g_cached_symbols(replacements)
    symbols = Set{Symbol}()
    for ex in replacements
        Meta.isexpr(ex, :(=), 2) && ex.args[1] isa Symbol && push!(symbols, ex.args[1])
    end
    symbols
end

function p2g_has_symbol(expr, symbols=nothing)
    if expr isa Symbol
        return symbols === nothing || expr in symbols
    elseif expr isa Expr
        return any(arg -> p2g_has_symbol(arg, symbols), expr.args)
    else
        return false
    end
end

function p2g_simple_factor(expr)
    expr isa Union{Symbol, Number, QuoteNode} && return true
    if Meta.isexpr(expr, :call, 2) && expr.args[1] in (:-, :+)
        return p2g_simple_factor(expr.args[2])
    end
    false
end

function p2g_product_expr(args::Vector)
    length(args) == 1 && return only(args)
    Expr(:call, :*, args...)
end

function hoist_p2g_rhs!(hoist_exprs::Vector, inner_symbols::Set{Symbol}, expr)
    MacroTools.postwalk(expr) do ex
        if Meta.isexpr(ex, :call) && first(ex.args) === :*
            return hoist_p2g_product_runs!(hoist_exprs, inner_symbols, Any[ex.args[2:end]...])
        end
        ex
    end
end

function hoist_p2g_product_runs!(hoist_exprs::Vector, inner_symbols::Set{Symbol}, args::Vector)
    newargs = Any[]
    run = Any[]
    for arg in args
        if p2g_simple_factor(arg) && !p2g_has_symbol(arg, inner_symbols)
            push!(run, arg)
        else
            append_p2g_hoisted_run!(newargs, hoist_exprs, run)
            empty!(run)
            push!(newargs, arg)
        end
    end
    append_p2g_hoisted_run!(newargs, hoist_exprs, run)
    p2g_product_expr(newargs)
end

function append_p2g_hoisted_run!(newargs::Vector, hoist_exprs::Vector, args)
    if length(args) > 1 && all(p2g_simple_factor, args) && any(p2g_has_symbol, args)
        sym = gensym(:p2g_rhs)
        rhs = p2g_product_expr(Any[args...])
        push!(hoist_exprs, :($sym = $rhs))
        push!(newargs, sym)
    else
        append!(newargs, args)
    end
    newargs
end

function p2g_sum_add_expr(lhs, index, rhs)
    Meta.isexpr(lhs, :ref, 2) || error("@P2G: invalid resolved LHS in `@∑` equation: $lhs")
    array, _ = lhs.args
    :(Tesserae.add!($array, $index, $rhs))
end

struct P2GSpGridStorageIndex
    i::Int
end

Base.@propagate_inbounds p2g_write_index(grid, i) = i
Base.@propagate_inbounds p2g_write_index(grid::Grid, i::CartesianIndex) = LinearIndices(grid)[i]
Base.@propagate_inbounds p2g_write_index(grid::SpGrid, i::CartesianIndex) = p2g_write_index(grid, get_spinds(grid)[Tuple(i)...])
# `@boundscheck` here means debug-mode only, not "always": the transfer macros
# wrap their generated body in `@inbounds` outside `debug_mode`, and that
# propagates into this callee and elides the check. Calling `update_sparsity!`
# is the caller's obligation under that same `@inbounds` contract, and
# `supportnodes(::BasisWeight, ::SpGrid)` and `add!`'s `@debug checkbounds`
# already assert it on the debug path.
#
# Do not promote this to an unconditional check without re-measuring. Making it
# fire in release costs 5-9% on a 3D sequential `SpGrid` transfer and 11.6%
# threaded; accumulating a flag branchlessly and raising once per particle still
# costs ~5%. The only measured way back to zero is to give `SpArray.data` a
# leading scrap slot so an inactive node lands there instead of out of bounds.
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

# `zeroed` are the grid fields the transfer assigns rather than accumulates
# into, which have to hold zero before the first particle scatters. They are
# passed down instead of being zeroed by the caller so that the threaded path
# can fold them into the parallel region it already opens; every other path
# zeroes them here, on this thread, before anything reads the grid.
#
# It defaults to `()` so that callers with nothing to zero -- `@P2G_Matrix`,
# which zeroes its own matrix targets -- need not say so.

# CPU: sequential
function P2G(f, ::CPUDevice, ::Val{scheduler}, grid, particles, weights, ::Nothing, zeroed::Tuple=()) where {scheduler}
    scheduler == :nothing || @warn "@P2G: `ThreadPartition` must be given for threaded computation" maxlog=1

    fillzero_each!(zeroed)
    for p in eachindex(particles)
        @inline f(grid, particles, weights, p)
    end
end

# CPU: multi-threading
function P2G(f, ::CPUDevice, ::Val{scheduler}, grid, particles, weights, partition::ThreadPartition, zeroed::Tuple=()) where {scheduler}
    # `partitioned_foreach` runs the prologue on its sequential paths too, so
    # the only case left here is targets no memset can reach.
    prologue = fillzero_prologue(zeroed)
    prologue === nothing && fillzero_each!(zeroed)
    partitioned_foreach(strategy(partition), Val(scheduler); prologue) do region
        for p in particle_indices(partition, particles, region)
            @inline f(grid, particles, weights, p)
        end
    end
end

# GPU
@kernel function gpukernel_P2G(f, grid, @Const(particles), @Const(weights))
    p = @index(Global)
    @inline f(grid, particles, weights, p)
end
function P2G(f, device::GPUDevice, ::Val{scheduler}, grid, particles, weights, ::Nothing, zeroed::Tuple=()) where {scheduler}
    scheduler == :nothing || @warn "Multi-threading is disabled for GPU" maxlog=1
    fillzero_each!(zeroed)
    particles = particles isa QuadraturePoints ? parent(particles) : particles
    backend = get_backend(device)
    kernel = gpukernel_P2G(backend)
    kernel(f, hybrid(grid), particles, weights; ndrange=length(particles))
end

# `f` is only handed on, so it takes a type parameter to be specialized on.
G2P2G(f::F, device::CPUDevice, schedule, grid, particles, weights, partition, zeroed::Tuple=()) where {F} =
    P2G(f, device, schedule, grid, particles, weights, partition, zeroed)

# Unlike P2G, G2P2G writes interpolated and updated particle properties.
@kernel function gpukernel_G2P2G(f, grid, particles, @Const(weights))
    p = @index(Global)
    @inline f(grid, particles, weights, p)
end
function G2P2G(f, device::GPUDevice, ::Val{scheduler}, grid, particles, weights, ::Nothing, zeroed::Tuple=()) where {scheduler}
    scheduler == :nothing || @warn "Multi-threading is disabled for GPU" maxlog=1
    fillzero_each!(zeroed)
    particles = particles isa QuadraturePoints ? parent(particles) : particles
    backend = get_backend(device)
    kernel = gpukernel_G2P2G(backend)
    kernel(f, hybrid(grid), particles, weights; ndrange=length(particles))
end

function P2G_nosum_expr((grid,i), nosum_equations::Vector)
    scope = TransferScope([grid=>i])
    nosum_equations = map(eq -> resolve_equation(eq, scope), nosum_equations)

    nosum_exprs = map(nosum_equations) do eq
        (; lhs, rhs, op) = eq
        Expr(op, lhs, rhs)
    end
    Expr(:block, nosum_exprs...)
end

# The grid-only part of `@P2G`. Every equation here reads and writes only its
# own node, which makes it the same walk `@foreach` runs over the same grid, so
# it shares the CPU loops and the GPU kernels in foreach.jl.
#
# Below this many nodes the fork-join costs more than the loop it replaces:
# threading the walk pays several times over on a grid large enough to cover it,
# and is a loss under it -- a 129^2 node slice with a body this cheap measures 3x
# slower threaded than sequential. Node count stands in for work only because
# these bodies are a handful of arithmetic operations per node, which is why
# `@foreach`, whose bodies are arbitrary and whose `@threaded` is an explicit
# request, shares the walk but not the threshold.
const P2G_NOSUM_MIN_THREADED_LENGTH = 1 << 15

P2G_nosum(f::F, ::CPUDevice, schedule::Val, grid) where {F} =
    cpu_foreach_loop(f, schedule, grid, P2G_NOSUM_MIN_THREADED_LENGTH)

# The GPU path already parallelises, so it ignores the scheduler.
P2G_nosum(f, device::GPUDevice, ::Val, grid) = P2G_nosum(f, device, grid)

function P2G_nosum(f, device::GPUDevice, grid)
    backend = get_backend(device)
    if grid isa SpGrid
        spinds = get_spinds(grid)
        kernel = gpukernel_foreach_spgrid(backend)
        kernel(f, grid, spinds; ndrange=_spindex_ndrange(spinds))
    else
        kernel = gpukernel_foreach(backend)
        kernel(f, grid; ndrange=size(grid))
    end
end

# `macroname` is threaded through so every transfer macro reports its own name.
# `@G2P` takes no partition and passes `nothing`, which skips the partition
# checks exactly as before.
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
    # check device
    device = get_device(grid)
    @assert get_device(particles) == get_device(weights) == device
    check_partition_for_transfer(macroname, device, grid, weights, partition)
end

# ThreadPartition is a CPU scheduling aid. GPU P2G uses particle-parallel kernels
# and SpGrid sparsity is updated separately from particle positions.
check_partition_for_transfer(macroname, ::CPUDevice, grid, weights, ::Nothing) = nothing
check_partition_for_transfer(macroname, ::GPUDevice, grid, weights, ::Nothing) = nothing
function check_partition_for_transfer(macroname, ::GPUDevice, grid, weights, partition)
    error("$macroname: ThreadPartition is only used on CPU. Use partitionless $macroname on GPU.")
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
    b = basis(first(weights))
    if support_width(b) > blockwidth(strat)
        error("$macroname: Block size for `ThreadPartition` is too small for basis $b. Increase `block_size_log2=Val(...)` on the `CartesianMesh` to ensure block size is ≥ kernel support.")
    end
end

"""
    @G2P grid=>i particles=>p weights=>ip begin
        equations...
    end

Grid-to-particle transfer macro.
Based on the `parent => index` expressions, `a[index]` in `equations`
translates to `parent.a[index]`. This `index` can be replaced with
any other name.

# Examples
```julia
@G2P grid=>i particles=>p weights=>ip begin

    # Grid-to-particle transfer
    v[p] += @∑ w[ip] * (vⁿ[i] - v[i])
    ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
    x[p] += @∑ w[ip] * v[i] * Δt

    # Calculation on particle
    Δϵₚ = symmetric(∇v[p]) * Δt
    F[p]  = (I + ∇v[p]*Δt) * F[p]
    V[p]  = V⁰[p] * det(F[p])
    σ[p] += λ*tr(Δϵₚ)*I + 2μ*Δϵₚ # Linear elastic material

end
```

This expands to roughly the following code:

```julia
# Grid-to-particle transfer
for p in eachindex(particles)
    bw = weights[p]
    nodeindices = supportnodes(bw)
    Δvₚ = zero(eltype(particles.v))
    ∇vₚ = zero(eltype(particles.∇v))
    Δxₚ = zero(eltype(particles.x))
    for ip in eachindex(nodeindices)
        i = nodeindices[ip]
        Δvₚ += bw.w[ip] * (grid.vⁿ[i] - grid.v[i])
        ∇vₚ += grid.v[i] ⊗ bw.∇w[ip]
        Δxₚ += bw.w[ip] * grid.v[i] * Δt
    end
    particles.v[p] += Δvₚ
    particles.∇v[p] = ∇vₚ
    particles.x[p] += Δxₚ
end

# Calculation on particle
for p in eachindex(particles)
    Δϵₚ = symmetric(particles.∇v[p]) * Δt
    particles.F[p]  = (I + particles.∇v[p]*Δt) * particles.F[p]
    particles.V[p]  = particles.V⁰[p] * det(particles.F[p])
    particles.σ[p] += λ*tr(Δϵₚ)*I + 2μ*Δϵₚ # Linear elastic material
end
```

Use `\$(expr)` inside transfer equations to evaluate an outer expression once
before the generated transfer loops and use the captured value in the loop body.
For example, `\$Δt` captures the current value of `Δt`.

!!! warning
    In `@G2P`, `Calculation on particles` part must be placed after
    `Grid-to-particle transfer` part.
"""
macro G2P(args...)
    schedule, grid_i, particles_p, weights_ip, _, equations = parse_transfer_macro_args("@G2P", args, false)
    G2P_expr(schedule, grid_i, particles_p, weights_ip, equations)
end

function G2P_expr(schedule::QuoteNode, grid_i::Expr, particles_p::Expr, weights_ip::Expr, equations::Expr)
    G2P_expr(schedule, unpair(grid_i), unpair(particles_p), unpair(weights_ip), parse_transfer_program(equations))
end

function G2P_expr(schedule::QuoteNode, (grid,i), (particles,p), (weights,ip), program::TransferProgram)
    sum_equations, nosum_equations = split_sum_equations(program, "@G2P")

    code = quote
        Tesserae.check_transfer_arguments("@G2P", $grid, $particles, $weights, nothing)
    end

    if !isempty(program.equations)
        body = G2P_sum_expr((grid,i), (particles,p), (weights,ip), sum_equations, nosum_equations)
        if !DEBUG
            body = :(@inbounds $body)
        end
        code = quote
            $code
            Tesserae.G2P(($grid, $particles, $weights, $p) -> $body, Tesserae.get_device($grid), Val($schedule), $(narrowed_grid_expr(grid, program.equations, i)), $particles, $weights)
        end
    end

    code = interpolate_transfer_values(code, program)
    esc(prettify(code; lines=true, alias=false))
end

# CPU: sequential & multi-threading
function G2P(f, ::CPUDevice, ::Val{scheduler}, grid, particles, weights) where {scheduler}
    tforeach(eachindex(particles), scheduler) do p
        @inline f(grid, particles, weights, p)
    end
end

# GPU
@kernel function gpukernel_G2P(f, @Const(grid), particles, @Const(weights))
    p = @index(Global)
    @inline f(grid, particles, weights, p)
end
function G2P(f, device::GPUDevice, ::Val{scheduler}, grid, particles, weights) where {scheduler}
    scheduler == :nothing || @warn "Multi-threading is disabled for GPU" maxlog=1
    particles = particles isa QuadraturePoints ? parent(particles) : particles
    backend = get_backend(device)
    kernel = gpukernel_G2P(backend)
    kernel(f, grid, particles, weights; ndrange=length(particles))
end

function G2P_sum_expr((grid,i), (particles,p), (weights,ip), sum_equations::Vector, nosum_equations::Vector, binding::SupportWindowBinding=SupportWindowBinding())
    (; window) = binding

    code = Expr(:block)
    scope = TransferScope([grid=>i, particles=>p, TrailingIndexed(weights, p)=>ip]; cache=true)

    if !isempty(sum_equations)
        sum_equations = resolve_sum_equations(sum_equations, scope, "@G2P", p)
        particle_replacements = cached_replacements(scope, p)
        inner_replacements = cached_replacements(scope, i, ip)

        inits = []
        saves = []
        sum_exprs = Any[]
        for eq in sum_equations
            (; lhs, rhs, op) = eq
            tmp = Symbol(lhs, :_p)
            push!(inits, :($tmp = zero(eltype($(remove_indexing(lhs))))))
            push!(saves, Expr(op, lhs, tmp))
            push!(sum_exprs, :($tmp += $rhs))
        end

        code = quote
            $(particle_replacements...)
            $(inits...)
            $(support_window_exprs(binding, weights, p)...)
            for $ip in eachindex($window)
                $i = Tesserae.transfer_nodeindex($grid, $window, $ip)
                $(inner_replacements...)
                $(sum_exprs...)
            end
            $(saves...)
        end
    end

    if !isempty(nosum_equations)
        nosum_scope = TransferScope([grid=>i, particles=>p, TrailingIndexed(weights, p)=>ip])
        nosum_equations = map(eq -> Expr(eq.op, resolve_refs(eq.lhs, nosum_scope), resolve_refs(eq.rhs, nosum_scope)), nosum_equations)
        code = quote
            $code
            $(nosum_equations...)
        end
    end

    code
end

"""
    @G2P2G grid=>i particles=>p weights=>ip [partition] begin
        equations...
    end

Combined grid-to-particle and particle-to-grid transfer macro.

Allows both [`@G2P`](@ref) (interpolation from grid to particles) and [`@P2G`](@ref) (scattering from particles to grid)
to be performed in a single loop over particles, avoiding repeated traversals.

# Examples
```julia
@G2P2G grid=>i particles=>p weights=>ip begin
    # G2P
    ∇v[p] = @∑ v[i] ⊗ ∇w[ip]

    # Particle update
    F[p] = (I + ∇v[p]*Δt) * F[p]
    σ[p] = cauchy_stress(F[p])

    # P2G
    f[i] = @∑ -V[p] * σ[p] * ∇w[ip]
end
```
"""
macro G2P2G(args...)
    G2P2G_expr(parse_transfer_macro_args("@G2P2G", args, true)...)
end

function G2P2G_expr(schedule::QuoteNode, grid_i::Expr, particles_p::Expr, weights_ip::Expr, partition, equations::Expr)
    G2P2G_expr(schedule, unpair(grid_i), unpair(particles_p), unpair(weights_ip), partition, parse_transfer_program(equations))
end

struct G2P2GStages
    g2p_sum::Vector{TransferEquation}
    p2g_sum::Vector{TransferEquation}
    g2p_nosum::Vector{TransferEquation}
    p2g_nosum::Vector{TransferEquation}
end

function split_g2p2g_stages(program::TransferProgram, i, p)
    equations_g2p_sum = TransferEquation[]
    equations_p2g_sum = TransferEquation[]
    equations_g2p_nosum = TransferEquation[]
    equations_p2g_nosum = TransferEquation[]
    precedence = 1
    for eq in program.equations
        if is_sum(eq)
            @capture(eq.lhs, A_[index_]) || error("@G2P2G: invalid LHS in `@∑` equation: $(eq.lhs)")
            if index == p
                precedence == 1 || error("@G2P2G: particle `@∑` equations must come before particle updates and grid-scattering equations")
                push!(equations_g2p_sum, eq)
            elseif index == i
                precedence ≤ 3 || error("@G2P2G: grid `@∑` equations must come before grid-only equations")
                push!(equations_p2g_sum, eq)
                precedence = 3
            else
                error("@G2P2G: wrong index in LHS equation, $(eq.lhs)")
            end
        else
            if precedence in (1, 2)
                push!(equations_g2p_nosum, eq)
                precedence = 2
            elseif precedence in (3, 4)
                push!(equations_p2g_nosum, eq)
                precedence = 4
            else
                error("unreachable")
            end
        end
    end
    G2P2GStages(equations_g2p_sum, equations_p2g_sum, equations_g2p_nosum, equations_p2g_nosum)
end

function G2P2G_expr(schedule::QuoteNode, (grid,i), (particles,p), (weights,ip), partition, program::TransferProgram)
    stages = split_g2p2g_stages(program, i, p)

    code = quote
        Tesserae.check_transfer_arguments("@G2P2G", $grid, $particles, $weights, $partition)
    end
    body = Expr(:block)
    binding = SupportWindowBinding()

    if !isempty(stages.g2p_sum) || !isempty(stages.g2p_nosum)
        expr = G2P_sum_expr((grid,i), (particles,p), (weights,ip), stages.g2p_sum, stages.g2p_nosum, binding)
        body = quote
            $body
            $expr
        end
    end

    zeroed = Expr(:tuple)
    if !isempty(stages.p2g_sum)
        # `G2P_sum_expr` binds the basis weight only when it has `@∑` equations,
        # so this half must load it itself otherwise.
        p2g_binding = isempty(stages.g2p_sum) ? binding : SupportWindowBinding(binding; load=false)
        zeroed, expr = P2G_sum_expr(schedule, (grid,i), (particles,p), (weights,ip), stages.p2g_sum, p2g_binding)
        body = quote
            $body
            $expr
        end
    end

    if !DEBUG
        body = :(@inbounds $body)
    end
    particle_equations = vcat(collect(stages.g2p_sum), collect(stages.g2p_nosum), collect(stages.p2g_sum))
    code = quote
        $code
        Tesserae.G2P2G(($grid, $particles, $weights, $p) -> $body, Tesserae.get_device($grid), Val($schedule), $(narrowed_grid_expr(grid, particle_equations, i)), $particles, $weights, $partition, $zeroed)
    end

    if !isempty(stages.p2g_nosum)
        body = P2G_nosum_expr((grid,i), stages.p2g_nosum)
        if !DEBUG
            body = :(@inbounds $body)
        end
        code = quote
            $code
            Tesserae.P2G_nosum(($grid, $i) -> $body, Tesserae.get_device($grid), Val($schedule), $(narrowed_grid_expr(grid, stages.p2g_nosum, i)))
        end
    end

    code = interpolate_transfer_values(code, program)
    esc(prettify(code; lines=true, alias=false))
end

####################
# Helper functions #
####################

function unpair(ex)
    if @capture(ex, lhs_Symbol => rhs_Symbol)
        return (lhs, rhs)
    elseif @capture(ex, lhs_Symbol => (rhs1_Symbol,rhs2_Symbol))
        return lhs, (rhs1, rhs2)
    elseif @capture(ex, (lhs1_Symbol,lhs2_Symbol) => (rhs1_Symbol,rhs2_Symbol))
        return (lhs1, lhs2), (rhs1, rhs2)
    else
        error("invalid expression, $ex")
    end
end

function has_sum_macro(expr)
    has_sum = Ref(false)
    MacroTools.postwalk(expr) do ex
        if Meta.isexpr(ex, :macrocall, 2) && (ex.args[1]==Symbol("@∑") || ex.args[1]==Symbol("@Σ"))
            has_sum[] = true
        end
        ex
    end
    has_sum[]
end

function parse_transfer_program(expr::Expr)
    expr = MacroTools.prewalk(MacroTools.rmlines, expr)
    @capture(expr, begin exprs__ end) || error("expected a `begin ... end` block, got $expr")
    interpolations = Pair{Symbol, Any}[]
    equations = map(exprs) do ex
        dict = MacroTools.trymatch(Expr(:op_, :lhs_, :rhs_), ex)
        dict === nothing && error("wrong expression: $ex")
        lhs, rhs, op = dict[:lhs], dict[:rhs], dict[:op]
        has_transfer_interpolation(lhs) && error("transfer interpolation with `\$` is only allowed on the RHS, got LHS `$lhs`")
        rhs = extract_transfer_interpolations(rhs, interpolations)
        if @capture(rhs, @∑ eq_)
            (op == :(=) || op == :(+=) || op == :(-=)) || error("@∑ is only allowed on the RHS of assignments with `=`, `+=`, or `-=`, got $ex")
            return TransferEquation(:sum, lhs, eq, op)
        end
        has_sum_macro(rhs) && error("@∑ must appear alone as the entire RHS expression, got $ex")
        TransferEquation(:assign, lhs, rhs, op)
    end
    TransferProgram(equations, interpolations)
end

function has_transfer_interpolation(expr)
    Meta.isexpr(expr, :$, 1) && return true
    expr isa Expr || return false
    any(has_transfer_interpolation, expr.args)
end

function extract_transfer_interpolations(expr, interpolations)
    if Meta.isexpr(expr, :$, 1)
        captured = gensym(:transfer_interp)
        push!(interpolations, captured => only(expr.args))
        return captured
    elseif expr isa Expr
        return Expr(expr.head, map(arg -> extract_transfer_interpolations(arg, interpolations), expr.args)...)
    else
        return expr
    end
end

function interpolate_transfer_values(code, program::TransferProgram)
    isempty(program.interpolations) && return code
    bindings = map(program.interpolations) do captured_rhs
        captured, rhs = captured_rhs
        Expr(:(=), captured, rhs)
    end
    Expr(:let, Expr(:block, bindings...), code)
end

function resolve_refs(expr, scope::TransferScope)
    MacroTools.postwalk(expr) do ex
        if @capture(ex, x_[i_]) && haskey(scope.bindings, i)
            parent = scope.bindings[i]
            if parent isa TrailingIndexed
                if scope.replacements === nothing
                    return :($(parent.parent).$x[$i, $(parent.trailing)])
                end
                viewsym = Symbol(:(view($(parent.parent).$x, $(parent.trailing))))
                push_unique!(scope.replacements[parent.trailing],
                             :($viewsym = Tesserae.weight_prop_view($(parent.parent), Val($(QuoteNode(x))), $(parent.trailing))))
                resolved = :($viewsym[$i])
                sym = Symbol(:($(parent.parent).$x[$i]))
            else
                resolved = :($parent.$x[$i])
                sym = Symbol(resolved)
            end
            scope.replacements === nothing && return resolved
            push_unique!(scope.replacements[i], :($sym = $resolved))
            return sym
        end
        ex
    end
end

function remove_indexing(expr)
    MacroTools.postwalk(expr) do ex
        @capture(ex, x_[i__]) && return x
        ex
    end
end

