# -----------------------------------------------------------------------------
#  @P2G
# -----------------------------------------------------------------------------

# ---- @P2G ----

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
        zeroed, body = P2G_sum_expr((grid,i), (particles,p), (weights,ip), sum_equations)
        if !DEBUG
            body = :(@inbounds $body)
        end
        particle_body = :(($grid, $particles, $weights, $p) -> $body)
        # The tile lowering is reached only through a partition; without one the
        # call below passes a literal `nothing`, so no tile-taking method applies.
        bodies = if partition === nothing
            particle_body
        else
            tile_names, tile_args, tile_body = P2G_tile_sum_expr((grid,i), (particles,p), (weights,ip), sum_equations)
            if !DEBUG
                tile_body = :(@inbounds $tile_body)
            end
            names_val = :(Val($(Expr(:tuple, map(QuoteNode, tile_names)...))))
            :(Tesserae.P2GBodies($particle_body,
                                 ($(tile_args...), $grid, $particles, $weights, $p) -> $tile_body,
                                 $names_val))
        end
        # Bound before the `do w` block: `w` is that block's parameter and would
        # shadow a user variable of the same name that a non-`@∑` equation reads.
        args = [bodies, :(Tesserae.get_device($grid)), :(Val($schedule)),
                narrowed_grid_expr(grid, sum_equations, i), particles, :w, partition, zeroed]
        bind = nothing
        if !isempty(nosum_equations)
            @gensym nodebody nodegrid
            bind = nosum_binding_expr(nodebody, nodegrid, grid, i, nosum_equations)
            append!(args, (nodebody, nodegrid))
        end
        call = isempty(nosum_equations) ? :(Tesserae.P2G($(args...))) : :(Tesserae.P2G_halves($(args...)))
        code = quote
            $code
            $bind
            Tesserae.select_weights($weights) do w
                $call
            end
        end
    elseif !isempty(nosum_equations)
        @gensym nodebody nodegrid
        code = quote
            $code
            $(nosum_binding_expr(nodebody, nodegrid, grid, i, nosum_equations))
            Tesserae.P2G_nosum($nodebody, Tesserae.get_device($grid), Val($schedule), $nodegrid)
        end
    end

    code = interpolate_transfer_values(code, program)
    esc(prettify(code; lines=true, alias=false))
end

# The two `@P2G` lowerings must resolve the same equations identically and differ
# only in what they do with the resolved RHS, so they share the resolution here.
function p2g_sum_scope((grid,i), (particles,p), (weights,ip), sum_equations::Vector, window,
                       cols::Union{WeightColumnsBinding,Nothing})
    cols = something(cols, WeightColumnsBinding(collect_transfer_refs(sum_equations, ip)))
    scope = TransferScope([grid=>i, particles=>p, TrailingIndexed(weights, p, particles, grid, window, cols)=>ip]; cache=true)
    equations = resolve_sum_equations(sum_equations, scope, "@P2G", i)
    particle_replacements = cached_replacements(scope, p)
    inner_replacements = cached_replacements(scope, i, ip)
    (; scope, equations, particle_replacements, inner_replacements,
       inner_symbols = p2g_cached_symbols(inner_replacements))
end

function P2G_sum_expr((grid,i), (particles,p), (weights,ip), sum_equations::Vector,
                      binding::SupportWindowBinding=SupportWindowBinding(),
                      cols::Union{WeightColumnsBinding,Nothing}=nothing)
    @gensym gridwriteindex
    (; window) = binding

    (; equations, particle_replacements, inner_replacements, inner_symbols) =
        p2g_sum_scope((grid,i), (particles,p), (weights,ip), sum_equations, window, cols)

    fillzero_targets = Any[]
    hoist_exprs = Any[]
    sum_exprs = Any[]
    for eq in equations
        (; lhs, rhs, op) = eq
        op == :(=)  && push_unique!(fillzero_targets, remove_indexing(lhs))
        op == :(-=) && (rhs = :(-$rhs))
        rhs = hoist_p2g_rhs!(hoist_exprs, inner_symbols, rhs)
        push!(sum_exprs, p2g_sum_add_expr(lhs, gridwriteindex, rhs))
    end

    body = quote
        $(support_window_exprs(binding, weights, particles, p, grid)...)
        $(particle_replacements...)
        $(hoist_exprs...)
        for $ip in eachindex($window)
            $i = Tesserae.transfer_nodeindex($grid, $window, $ip)
            $gridwriteindex = Tesserae.p2g_write_index($grid, $i)
            $(inner_replacements...)
            $(sum_exprs...)
        end
    end

    Expr(:tuple, fillzero_targets...), body
end

# The tile lowering of the same equations: writes go through `tile_add!` at the
# node's local slot, every read stays as in the particle-parallel body.
function P2G_tile_sum_expr((grid,i), (particles,p), (weights,ip), sum_equations::Vector)
    @gensym tile origin sideval tilelenval tileslot offsets
    binding = SupportWindowBinding()
    (; window) = binding

    (; scope, equations, particle_replacements, inner_replacements, inner_symbols) =
        p2g_sum_scope((grid,i), (particles,p), (weights,ip), sum_equations, window, nothing)

    fieldnames = Symbol[]
    hoist_exprs = Any[]
    sum_exprs = Any[]
    for eq in equations
        (; lhs, rhs, op) = eq
        name = tile_field_name(lhs)
        name in fieldnames || push!(fieldnames, name)
        op == :(-=) && (rhs = :(-$rhs))
        rhs = hoist_p2g_rhs!(hoist_exprs, inner_symbols, rhs)
        slot = findfirst(==(name), fieldnames)
        push!(sum_exprs, :(Tesserae.tile_add!($tile, $tilelenval, $offsets[$slot], $tileslot, $rhs)))
    end

    # The tile writes address the shared tile through the local slot alone, so the
    # node index is only needed when an equation mentions `i`.
    uses_i = !isempty(scope.replacements[i]) || any(ex -> expr_contains_symbol(ex, i), sum_exprs)
    node_exprs = uses_i ? (:($i = Tesserae.transfer_nodeindex($grid, $window, $ip)),) : ()
    names_val = :(Val($(Expr(:tuple, map(QuoteNode, fieldnames)...))))
    body = quote
        $offsets = Tesserae.tile_offsets($grid, $names_val)
        $(support_window_exprs(binding, weights, particles, p, grid)...)
        $(particle_replacements...)
        $(hoist_exprs...)
        for $ip in eachindex($window)
            $(node_exprs...)
            $tileslot = Tesserae.tile_slot($window[$ip], $origin, $sideval)
            $(inner_replacements...)
            $(sum_exprs...)
        end
    end

    args = (tile, origin, sideval, tilelenval)
    fieldnames, args, body
end

function expr_contains_symbol(ex, sym::Symbol)
    found = false
    MacroTools.postwalk(ex) do x
        x === sym && (found = true)
        x
    end
    found
end

# `resolve_sum_equations` has already rewritten the LHS, so `<grid>.<name>[i]` is
# the only shape reaching here.
function tile_field_name(lhs)
    @capture(lhs, x_.name_[i_]) || error("@P2G: cannot derive the scattered field from `$(lhs)`")
    name
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

# `zeroed` are the grid fields the transfer assigns rather than accumulates into.
# They are passed down instead of being zeroed by the caller so that the threaded
# path can fold them into the parallel region it already opens.
function P2G(f, ::CPUDevice, ::Val{scheduler}, grid, particles, weights, ::Nothing, zeroed::Tuple=()) where {scheduler}
    scheduler == :nothing || @warn "@P2G: `ThreadPartition` must be given for threaded computation" maxlog=1

    fillzero_each!(zeroed)
    for p in eachindex(particles)
        @inline f(grid, particles, weights, p)
    end
end

function P2G(f, device::CPUDevice, schedule::Val, grid, particles, weights, partition::ThreadPartition, zeroed::Tuple=())
    p2g_region(f, device, schedule, grid, particles, weights, partition, zeroed, nothing)
end

# `epilogue` is how `@P2G`'s grid-node half rides this region instead of forking
# one of its own; see `P2G_halves`.
function p2g_region(f, ::CPUDevice, ::Val{scheduler}, grid, particles, weights,
                    partition::ThreadPartition, zeroed::Tuple, epilogue::E) where {scheduler, E}
    # `partitioned_foreach` runs the prologue on its sequential paths too, so
    # the only case left here is targets no memset can reach.
    prologue = fillzero_prologue(zeroed)
    prologue === nothing && fillzero_each!(zeroed)
    partitioned_foreach(strategy(partition), Val(scheduler); prologue, epilogue) do region
        for p in particle_indices(partition, particles, region)
            @inline f(grid, particles, weights, p)
        end
    end
end

# Only the threaded CPU path fuses the two halves. Keeping the rest as separate
# calls makes the GPU tile fallback safe: the node half is discharged here, not
# inside a method that may return early.
function P2G_halves(f, device, schedule, grid, particles, weights, partition, zeroed,
                    nodebody::N, nodegrid) where {N}
    P2G(f, device, schedule, grid, particles, weights, partition, zeroed)
    P2G_nosum(nodebody, device, schedule, nodegrid)
end

function P2G_halves(f, device::CPUDevice, schedule::Val, grid, particles, weights,
                    partition::ThreadPartition, zeroed::Tuple, nodebody::N, nodegrid) where {N}
    epilogue = (nworkers, w) -> foreach_worker_loop(nodebody, device, nodegrid, nworkers, w)
    p2g_region(f, device, schedule, grid, particles, weights, partition, zeroed, epilogue)
end

# Only the block-scheduled GPU path uses the tile body.
P2G_halves(bodies::P2GBodies, device::CPUDevice, schedule::Val, grid, particles, weights,
           partition::ThreadPartition, zeroed::Tuple, nodebody, nodegrid) =
    P2G_halves(bodies.particle, device, schedule, grid, particles, weights, partition, zeroed, nodebody, nodegrid)

# Bodies fall back to the particle-parallel lowering everywhere except the
# block-scheduled GPU path below. One method per signature, so each is strictly
# more specific than its `f`-taking sibling.
P2G(bodies::P2GBodies, device::CPUDevice, schedule::Val, grid, particles, weights, ::Nothing, zeroed::Tuple=()) =
    P2G(bodies.particle, device, schedule, grid, particles, weights, nothing, zeroed)
P2G(bodies::P2GBodies, device::CPUDevice, schedule::Val, grid, particles, weights, partition::ThreadPartition, zeroed::Tuple=()) =
    P2G(bodies.particle, device, schedule, grid, particles, weights, partition, zeroed)
P2G(bodies::P2GBodies, device::GPUDevice, schedule::Val, grid, particles, weights, ::Nothing, zeroed::Tuple=()) =
    P2G(bodies.particle, device, schedule, grid, particles, weights, nothing, zeroed)

# Shared so the fused and separate routes hand on the same pair.
function nosum_binding_expr(nodebody, nodegrid, grid, i, nosum_equations::Vector)
    body = P2G_nosum_expr((grid,i), nosum_equations)
    DEBUG || (body = :(@inbounds $body))
    quote
        local $nodebody = ($grid, $i) -> $body
        local $nodegrid = $(narrowed_grid_expr(grid, nosum_equations, i))
    end
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
