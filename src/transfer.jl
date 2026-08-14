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

struct TransferScope
    bindings::Dict{Any,Symbol}
    replacements::Union{Nothing,Dict{Any,Vector{Expr}}}
end

function TransferScope(maps::Vector{<: Pair}; cache::Bool=false)
    bindings = Dict{Any,Symbol}()
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
# the basis weight and its support nodes must be bound once and shared by both
# halves. The two symbols travel together with a `load` flag: whichever half
# holds `load=true` emits the binding statements, the other just uses them.
struct BasisWeightBinding
    bw::Symbol
    gridindices::Symbol
    load::Bool
end
BasisWeightBinding() = BasisWeightBinding(gensym(:bw), gensym(:gridindices), true)
BasisWeightBinding(binding::BasisWeightBinding; load::Bool) = BasisWeightBinding(binding.bw, binding.gridindices, load)

function basis_weight_exprs(binding::BasisWeightBinding, grid, weights, p)
    binding.load || return ()
    (:($(binding.bw) = $weights[$p]), :($(binding.gridindices) = supportnodes($(binding.bw), $grid)))
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
macro P2G(grid_i, particles_p, weights_ip, equations)
    P2G_expr(QuoteNode(:nothing), grid_i, particles_p, weights_ip, nothing, equations)
end
macro P2G(grid_i, particles_p, weights_ip, partition, equations)
    P2G_expr(QuoteNode(:nothing), grid_i, particles_p, weights_ip, partition, equations)
end
macro P2G(schedule::QuoteNode, grid_i, particles_p, weights_ip, equations)
    P2G_expr(schedule, grid_i, particles_p, weights_ip, nothing, equations)
end
macro P2G(schedule::QuoteNode, grid_i, particles_p, weights_ip, partition, equations)
    P2G_expr(schedule, grid_i, particles_p, weights_ip, partition, equations)
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
        pre, body = P2G_sum_expr((grid,i), (particles,p), (weights,ip), sum_equations)
        if !DEBUG
            body = :(@inbounds $body)
        end
        code = quote
            $code
            $pre
            Tesserae.P2G(($grid, $particles, $weights, $p) -> $body, Tesserae.get_device($grid), Val($schedule), $grid, $particles, $weights, $partition)
        end
    end

    if !isempty(nosum_equations)
        body = P2G_nosum_expr((grid,i), nosum_equations)
        if !DEBUG
            body = :(@inbounds $body)
        end
        code = quote
            $code
            Tesserae.P2G_nosum(($grid, $i) -> $body, Tesserae.get_device($grid), $grid)
        end
    end

    code = interpolate_transfer_values(code, program)
    esc(prettify(code; lines=true, alias=false))
end

function P2G_sum_expr((grid,i), (particles,p), (weights,ip), sum_equations::Vector, binding::BasisWeightBinding=BasisWeightBinding())
    @gensym gridwriteindex
    (; bw, gridindices) = binding

    scope = TransferScope([grid=>i, particles=>p, bw=>ip]; cache=true)
    sum_equations = resolve_sum_equations(sum_equations, scope, "@P2G", i)
    particle_replacements = cached_replacements(scope, p)
    inner_replacements = cached_replacements(scope, i, ip)
    inner_symbols = p2g_cached_symbols(inner_replacements)

    fillzeros = Any[]
    hoist_exprs = Any[]
    sum_exprs = Any[]
    for eq in sum_equations
        (; lhs, rhs, op) = eq
        op == :(=)  && push_unique!(fillzeros, :(Tesserae.fillzero!($(remove_indexing(lhs)))))
        op == :(-=) && (rhs = :(-$rhs))
        rhs = hoist_p2g_rhs!(hoist_exprs, inner_symbols, rhs)
        push!(sum_exprs, p2g_sum_add_expr(lhs, gridwriteindex, rhs))
    end

    body = quote
        $(particle_replacements...)
        $(hoist_exprs...)
        $(basis_weight_exprs(binding, grid, weights, p)...)
        for $ip in eachindex($gridindices)
            $i = $gridindices[$ip]
            $gridwriteindex = Tesserae.p2g_write_index($grid, $i)
            $(inner_replacements...)
            $(sum_exprs...)
        end
    end

    Expr(:block, fillzeros...), body
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

# CPU: sequential
function P2G(f, ::CPUDevice, ::Val{scheduler}, grid, particles, weights, ::Nothing) where {scheduler}
    scheduler == :nothing || @warn "@P2G: `ThreadPartition` must be given for threaded computation" maxlog=1

    for p in eachindex(particles)
        @inline f(grid, particles, weights, p)
    end
end

# CPU: multi-threading
function P2G(f, ::CPUDevice, ::Val{scheduler}, grid, particles, weights, partition::ThreadPartition) where {scheduler}
    for group in threadsafe_groups(partition)
        tforeach(group, scheduler) do region
            for p in particle_indices(partition, particles, region)
                @inline f(grid, particles, weights, p)
            end
        end
    end
end

# GPU
@kernel function gpukernel_P2G(f, grid, @Const(particles), @Const(weights))
    p = @index(Global)
    f(grid, particles, weights, p)
end
function P2G(f, device::GPUDevice, ::Val{scheduler}, grid, particles, weights, ::Nothing) where {scheduler}
    scheduler == :nothing || @warn "Multi-threading is disabled for GPU" maxlog=1
    particles = particles isa QuadraturePoints ? parent(particles) : particles
    backend = get_backend(device)
    kernel = gpukernel_P2G(backend)
    kernel(f, hybrid(grid), particles, weights; ndrange=length(particles))
end

G2P2G(f, device::CPUDevice, schedule, grid, particles, weights, partition) =
    P2G(f, device, schedule, grid, particles, weights, partition)

# Unlike P2G, G2P2G writes interpolated and updated particle properties.
@kernel function gpukernel_G2P2G(f, grid, particles, @Const(weights))
    p = @index(Global)
    f(grid, particles, weights, p)
end
function G2P2G(f, device::GPUDevice, ::Val{scheduler}, grid, particles, weights, ::Nothing) where {scheduler}
    scheduler == :nothing || @warn "Multi-threading is disabled for GPU" maxlog=1
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

grid_node_indices(grid) = eachindex(grid)
grid_node_indices(grid::SpGrid) = activeindices(get_spinds(grid))

function P2G_nosum(f, ::CPUDevice, grid)
    @inbounds @simd for i in grid_node_indices(grid)
        @inline f(grid, i)
    end
end

function P2G_nosum(f, ::CPUDevice, grid::SpGrid)
    @inbounds for i in grid_node_indices(grid)
        @inline f(grid, i)
    end
end

@kernel function gpukernel_P2G_nosum(f, grid)
    i = @index(Global, Cartesian)
    f(grid, i)
end

@kernel function gpukernel_P2G_nosum_spgrid(f, grid, @Const(spinds))
    k = @index(Global)
    active, i = _active_spindex(spinds, k)
    if active
        @inbounds f(grid, i)
    end
end

function P2G_nosum(f, device::GPUDevice, grid)
    backend = get_backend(device)
    if grid isa SpGrid
        spinds = get_spinds(grid)
        kernel = gpukernel_P2G_nosum_spgrid(backend)
        kernel(f, grid, spinds; ndrange=_spindex_ndrange(spinds))
    else
        kernel = gpukernel_P2G_nosum(backend)
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
macro G2P(grid_i, particles_p, weights_ip, equations)
    G2P_expr(QuoteNode(:nothing), grid_i, particles_p, weights_ip, equations)
end
macro G2P(schedule::QuoteNode, grid_i, particles_p, weights_ip, equations)
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
            Tesserae.G2P(($grid, $particles, $weights, $p) -> $body, Tesserae.get_device($grid), Val($schedule), $grid, $particles, $weights)
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
    f(grid, particles, weights, p)
end
function G2P(f, device::GPUDevice, ::Val{scheduler}, grid, particles, weights) where {scheduler}
    scheduler == :nothing || @warn "Multi-threading is disabled for GPU" maxlog=1
    particles = particles isa QuadraturePoints ? parent(particles) : particles
    backend = get_backend(device)
    kernel = gpukernel_G2P(backend)
    kernel(f, grid, particles, weights; ndrange=length(particles))
end

function G2P_sum_expr((grid,i), (particles,p), (weights,ip), sum_equations::Vector, nosum_equations::Vector, binding::BasisWeightBinding=BasisWeightBinding())
    (; bw, gridindices) = binding

    code = Expr(:block)
    scope = TransferScope([grid=>i, particles=>p, bw=>ip]; cache=true)

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
            $(basis_weight_exprs(binding, grid, weights, p)...)
            for $ip in eachindex($gridindices)
                $i = $gridindices[$ip]
                $(inner_replacements...)
                $(sum_exprs...)
            end
            $(saves...)
        end
    end

    if !isempty(nosum_equations)
        nosum_scope = TransferScope([grid=>i, particles=>p, bw=>ip])
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
macro G2P2G(grid_i, particles_p, weights_ip, equations)
    G2P2G_expr(QuoteNode(:nothing), grid_i, particles_p, weights_ip, nothing, equations)
end
macro G2P2G(grid_i, particles_p, weights_ip, partition, equations)
    G2P2G_expr(QuoteNode(:nothing), grid_i, particles_p, weights_ip, partition, equations)
end
macro G2P2G(schedule::QuoteNode, grid_i, particles_p, weights_ip, equations)
    G2P2G_expr(schedule, grid_i, particles_p, weights_ip, nothing, equations)
end
macro G2P2G(schedule::QuoteNode, grid_i, particles_p, weights_ip, partition, equations)
    G2P2G_expr(schedule, grid_i, particles_p, weights_ip, partition, equations)
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
    binding = BasisWeightBinding()

    if !isempty(stages.g2p_sum) || !isempty(stages.g2p_nosum)
        expr = G2P_sum_expr((grid,i), (particles,p), (weights,ip), stages.g2p_sum, stages.g2p_nosum, binding)
        body = quote
            $body
            $expr
        end
    end

    if !isempty(stages.p2g_sum)
        # `G2P_sum_expr` binds the basis weight only when it has `@∑` equations,
        # so this half must load it itself otherwise.
        p2g_binding = isempty(stages.g2p_sum) ? binding : BasisWeightBinding(binding; load=false)
        pre, expr = P2G_sum_expr((grid,i), (particles,p), (weights,ip), stages.p2g_sum, p2g_binding)
        code = quote
            $code
            $pre
        end
        body = quote
            $body
            $expr
        end
    end

    if !DEBUG
        body = :(@inbounds $body)
    end
    code = quote
        $code
        Tesserae.G2P2G(($grid, $particles, $weights, $p) -> $body, Tesserae.get_device($grid), Val($schedule), $grid, $particles, $weights, $partition)
    end

    if !isempty(stages.p2g_nosum)
        body = P2G_nosum_expr((grid,i), stages.p2g_nosum)
        if !DEBUG
            body = :(@inbounds $body)
        end
        code = quote
            $code
            Tesserae.P2G_nosum(($grid, $i) -> $body, Tesserae.get_device($grid), $grid)
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
            resolved = :($parent.$x[$i])
            scope.replacements === nothing && return resolved
            sym = Symbol(resolved)
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

