using MacroTools

###############
# HybridArray #
###############

# HybridArray is used to handle atomic operations on the GPU.
# Since atomic operations do not support custom bitstypes such as `Tensor`,
# the data is flattened into a HybridArray.

struct HybridArray{T, N, A <: AbstractArray{T, N}, B <: AbstractArray, D <: AbstractDevice} <: AbstractArray{T, N}
    parent::A
    flat::B
    device::D # stored in advance to avoid the overhead of calling `get_device` in a loop.
end

Base.parent(A::HybridArray) = A.parent
flatten(A::HybridArray) =  A.flat
get_device(A::HybridArray) = A.device

Base.size(A::HybridArray) = size(parent(A))
Base.IndexStyle(::Type{<: HybridArray{<: Any, <: Any, A}}) where {A} = IndexStyle(A)

@inline function Base.getindex(A::HybridArray, I...)
    @boundscheck checkbounds(parent(A), I...)
    @inbounds parent(A)[I...]
end
@inline function Base.setindex!(A::HybridArray, v, I...)
    @boundscheck checkbounds(parent(A), I...)
    @inbounds parent(A)[I...] = v
    A
end

@inline add!(A::AbstractArray{T}, i, v::T) where {T} = (@_propagate_inbounds_meta; A[i] += v)
@inline add!(A::HybridArray{T}, i, v::T) where {T} = (@_propagate_inbounds_meta; _add!(get_device(A), A, i, v))
@inline _add!(::CPUDevice, A::HybridArray, i, v) = (@_propagate_inbounds_meta; add!(parent(A), i, v))
@inline function _add!(::GPUDevice, A::HybridArray, i, v::Number)
    @_propagate_inbounds_meta
    Atomix.@atomic parent(A)[i] += v
end
@inline function _add!(::GPUDevice, A::HybridArray, i, v::Union{Tensor, StaticArray})
    @_propagate_inbounds_meta
    data = Tuple(v)
    for j in eachindex(data)
        Atomix.@atomic flatten(A)[j,i] += data[j]
    end
end

flatten(A::AbstractArray{T}) where {T <: Number} = reshape(A, 1, size(A)...)
flatten(A::AbstractArray{T}) where {T <: Tensor} = reinterpret(reshape, eltype(T), A)

hybrid(A::AbstractArray{T}) where {T} = HybridArray(A, flatten(A), get_device(A))
hybrid(A::SpArray{T}) where {T} = HybridArray(A, flatten(A), get_device(A))
hybrid(A::StructArray) = StructArray(map(hybrid, StructArrays.components(A)))
hybrid(mesh::AbstractMesh) = mesh

struct TransferEquation
    kind::Symbol
    lhs::Any
    rhs::Any
    op::Symbol
end

is_sum(eq::TransferEquation) = eq.kind === :sum

struct TransferProgram
    equations::Vector{TransferEquation}
end

function split_sum_equations(program::TransferProgram, macroname::String)
    equations = program.equations
    issum = map(is_sum, equations)
    if !allequal(issum) && !issorted(issum; rev=true)
        error("$macroname: Equations without `@∑` must come after those with `@∑`")
    end
    equations[issum], equations[.!issum]
end

struct TransferBinding
    parent::Symbol
    index::Any
end

struct TransferScope
    bindings::Vector{TransferBinding}
    replacements::Union{Nothing, Vector{Vector{Expr}}}
end

function TransferScope(maps::Vector{<: Pair}; cache::Bool=false)
    bindings = map(maps) do map
        TransferBinding(map.first, map.second)
    end
    replacements = cache ? replacement_groups(length(bindings)) : nothing
    TransferScope(bindings, replacements)
end

uncached(scope::TransferScope) = TransferScope(scope.bindings, nothing)

function cached_replacements(scope::TransferScope, groups::Integer...)
    scope.replacements === nothing && error("reference cache is not enabled for this transfer scope")
    replacements(scope.replacements, groups...)
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

replacement_groups(n::Integer) = [Expr[] for _ in 1:n]

function push_unique_expr!(xs::Vector{Expr}, x::Expr)
    x in xs || push!(xs, x)
    xs
end

function append_unique_exprs!(dst::Vector{Expr}, src::Vector{Expr})
    for x in src
        push_unique_expr!(dst, x)
    end
    dst
end

function replacements(replaced::Vector{Vector{Expr}}, groups::Integer...)
    exprs = Expr[]
    for group in groups
        append_unique_exprs!(exprs, replaced[group])
    end
    exprs
end

function push_unique!(xs::Vector, x)
    x in xs || push!(xs, x)
    xs
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
@. grid.vⁿ = grid.mv / grid.m
@. grid.v  = grid.vⁿ + (grid.f / grid.m) * Δt
```

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
        Tesserae.check_arguments_for_P2G($grid, $particles, $weights, $partition)
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
        code = quote
            $code
            let
                $body
            end
        end
    end

    esc(prettify(code; lines=true, alias=false))
end

function P2G_sum_expr((grid,i), (particles,p), (weights,ip), sum_equations::Vector)
    @gensym bw gridindices

    scope = TransferScope([grid=>i, particles=>p, bw=>ip]; cache=true)
    sum_equations = resolve_sum_equations(sum_equations, scope, "@P2G", i)
    replaced = scope.replacements

    fillzeros = Any[]
    sum_exprs = Any[]
    for eq in sum_equations
        (; lhs, rhs, op) = eq
        op == :(=)  && push_unique!(fillzeros, :(Tesserae.fillzero!($(remove_indexing(lhs)))))
        op == :(-=) && (rhs = :(-$rhs))
        push!(sum_exprs, :(Tesserae.add!($(lhs.args...), $rhs)))
    end

    body = quote
        $(replaced[2]...)
        $bw = $weights[$p]
        $gridindices = supportnodes($bw, $grid)
        for $ip in eachindex($gridindices)
            $i = $gridindices[$ip]
            $(cached_replacements(scope, 1, 3)...)
            $(sum_exprs...)
        end
    end
    
    Expr(:block, fillzeros...), body
end

# CPU: sequential
function P2G(f, ::CPUDevice, ::Val{scheduler}, grid, particles, weights, ::Nothing) where {scheduler}
    scheduler == :nothing || @warn "@P2G: `ColorPartition` must be given for threaded computation" maxlog=1

    for p in eachindex(particles)
        @inline f(grid, particles, weights, p)
    end
end

# CPU: multi-threading
function P2G(f, ::CPUDevice, ::Val{scheduler}, grid, particles, weights, partition::ColorPartition{<: BlockStrategy}) where {scheduler}
    strat = strategy(partition)
    for group in colorgroups(strat)
        tforeach(group, scheduler) do blk
            for p in particle_indices_in(strat, blk)
                @inline f(grid, particles, weights, p)
            end
        end
    end
end
function P2G(f, ::CPUDevice, ::Val{scheduler}, grid, particles, weights, partition::ColorPartition{<: CellStrategy}) where {scheduler}
    strat = strategy(partition)
    for group in colorgroups(strat)
        tforeach(group, scheduler) do cell
            for p in 1:size(particles, 1)
                @inline f(grid, particles, weights, CartesianIndex(p, cell))
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
    backend = get_backend(device)
    kernel = gpukernel_P2G(backend)
    kernel(f, hybrid(grid), particles, weights; ndrange=length(particles))
    synchronize(backend)
end

function P2G_nosum_expr((grid,i), nosum_equations::Vector)
    scope = TransferScope([grid=>i])
    nosum_equations = map(nosum_equations) do eq
        eq = resolve_equation(eq, scope)
        TransferEquation(eq.kind, remove_indexing(eq.lhs), remove_indexing(eq.rhs), eq.op)
    end

    nosum_exprs = map(nosum_equations) do eq
        (; lhs, rhs, op) = eq
        if @capture(lhs, $grid.x_)
            :(@. $(Expr(op, lhs, rhs)))
        else # TODO: avoid intermediate allocation
            Expr(op, lhs, :(@. $rhs))
        end
    end
    Expr(:block, nosum_exprs...)
end

function check_arguments_for_P2G(grid, particles, weights, partition)
    get_mesh(grid) isa AbstractMesh || error("@P2G: grid must have a mesh")
    eltype(weights) <: BasisWeight || error("@P2G: invalid `BasisWeight`s, got type $(typeof(weights))")
    if grid isa SpGrid
        if length(propertynames(grid)) > 1
            isempty(get_data(getproperty(grid, 2))) && error("@P2G: SpGrid indices not activated")
        end
    end
    @assert length(particles) ≤ length(weights)
    if partition isa ColorPartition
        strat = strategy(partition)
        if strat isa BlockStrategy
            @assert nblocks(get_mesh(grid)) == nblocks(strat)
            if sum(length(particle_indices_in(strat, blk)) for blk in LinearIndices(nblocks(strat))) == 0
                error("@P2G: No particles assigned to any block in ColorPartition")
            end
            b = basis(first(weights))
            if kernel_support(b) > blockwidth(strat)
                error("@P2G: Block size for `ColorPartition` is too small for basis $b. Increase `block_size_log2=Val(...)` on the `CartesianMesh` to ensure block size is ≥ kernel support.")
            end
        end
    end
    # check device
    device = get_device(grid)
    @assert get_device(particles) == get_device(weights) == device
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
        Tesserae.check_arguments_for_G2P($grid, $particles, $weights)
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
    backend = get_backend(device)
    kernel = gpukernel_G2P(backend)
    kernel(f, grid, particles, weights; ndrange=length(particles))
    synchronize(backend)
end

function G2P_sum_expr((grid,i), (particles,p), (weights,ip), sum_equations::Vector, nosum_equations::Vector)
    @gensym bw gridindices

    code = Expr(:block)
    scope = TransferScope([grid=>i, particles=>p, bw=>ip]; cache=true)

    if !isempty(sum_equations)
        sum_equations = resolve_sum_equations(sum_equations, scope, "@G2P", p)
        replaced = scope.replacements

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
            $(replaced[2]...)
            $(inits...)
            $bw = $weights[$p]
            $gridindices = supportnodes($bw, $grid)
            for $ip in eachindex($gridindices)
                $i = $gridindices[$ip]
                $(cached_replacements(scope, 1, 3)...)
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
        Tesserae.check_arguments_for_G2P($grid, $particles, $weights)
        Tesserae.check_arguments_for_P2G($grid, $particles, $weights, $partition)
    end
    body = Expr(:block)

    if !isempty(stages.g2p_sum) || !isempty(stages.g2p_nosum)
        expr = G2P_sum_expr((grid,i), (particles,p), (weights,ip), stages.g2p_sum, stages.g2p_nosum)
        body = quote
            $body
            $expr
        end
    end

    if !isempty(stages.p2g_sum)
        pre, expr = P2G_sum_expr((grid,i), (particles,p), (weights,ip), stages.p2g_sum)
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
        Tesserae.P2G(($grid, $particles, $weights, $p) -> $body, Tesserae.get_device($grid), Val($schedule), $grid, $particles, $weights, $partition)
    end

    if !isempty(stages.p2g_nosum)
        body = P2G_nosum_expr((grid,i), stages.p2g_nosum)
        code = quote
            $code
            let
                $body
            end
        end
    end

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
    equations = map(exprs) do ex
        dict = MacroTools.trymatch(Expr(:op_, :lhs_, :rhs_), ex)
        dict === nothing && error("wrong expression: $ex")
        lhs, rhs, op = dict[:lhs], dict[:rhs], dict[:op]
        if @capture(rhs, @∑ eq_)
            (op == :(=) || op == :(+=) || op == :(-=)) || error("@∑ is only allowed on the RHS of assignments with `=`, `+=`, or `-=`, got $ex")
            return TransferEquation(:sum, lhs, eq, op)
        end
        has_sum_macro(rhs) && error("@∑ must appear alone as the entire RHS expression, got $ex")
        TransferEquation(:assign, lhs, rhs, op)
    end
    TransferProgram(equations)
end

function resolve_refs(expr, scope::TransferScope)
    MacroTools.postwalk(expr) do ex
        if @capture(ex, x_[i_])
            for (k, binding) in enumerate(scope.bindings)
                if i == binding.index
                    parent = binding.parent
                    resolved = :($parent.$x[$i])
                    scope.replacements === nothing && return resolved
                    sym = Symbol(resolved)
                    push_unique_expr!(scope.replacements[k], :($sym = $resolved))
                    return sym
                end
            end
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

function check_arguments_for_G2P(grid, particles, weights)
    get_mesh(grid) isa AbstractMesh || error("@G2P: grid must have a mesh")
    eltype(weights) <: BasisWeight || error("@G2P: invalid `BasisWeight`s, got type $(typeof(weights))")
    if grid isa SpGrid
        if length(propertynames(grid)) > 1
            isempty(get_data(getproperty(grid, 2))) && error("@G2P: SpGrid indices not activated")
        end
    end
    @assert length(particles) ≤ length(weights)
    # check device
    device = get_device(grid)
    @assert get_device(particles) == get_device(weights) == device
end
