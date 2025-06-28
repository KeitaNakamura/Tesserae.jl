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

mutable struct Equation
    issumeq::Bool
    lhs::Any
    rhs::Any
    op::Symbol
end

"""
    @P2G grid=>i particles=>p mpvalues=>ip [partition] begin
        equations...
    end

Particle-to-grid transfer macro.
Based on the `parent => index` expressions, `a[index]` in `equations`
translates to `parent.a[index]`. This `index` can be replaced with
any other name.

# Examples
```julia
@P2G grid=>i particles=>p mpvalues=>ip begin

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
    mp = mpvalues[p]
    nodeindices = neighboringnodes(mp)
    for ip in eachindex(nodeindices)
        i = nodeindices[ip]
        grid.m [i] += mp.w[ip] * particles.m[p]
        grid.mv[i] += mp.w[ip] * particles.m[p] * particles.v[p]
        grid.mv[i] += -particles.V[p] * particles.σ[p] * mp.∇w[ip]
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
macro P2G(grid_i, particles_p, mpvalues_ip, equations)
    P2G_expr(QuoteNode(:nothing), grid_i, particles_p, mpvalues_ip, nothing, equations)
end
macro P2G(grid_i, particles_p, mpvalues_ip, partition, equations)
    P2G_expr(QuoteNode(:nothing), grid_i, particles_p, mpvalues_ip, partition, equations)
end
macro P2G(schedule::QuoteNode, grid_i, particles_p, mpvalues_ip, equations)
    P2G_expr(schedule, grid_i, particles_p, mpvalues_ip, nothing, equations)
end
macro P2G(schedule::QuoteNode, grid_i, particles_p, mpvalues_ip, partition, equations)
    P2G_expr(schedule, grid_i, particles_p, mpvalues_ip, partition, equations)
end

function P2G_expr(schedule::QuoteNode, grid_i::Expr, particles_p::Expr, mpvalues_ip::Expr, partition, equations::Expr)
    P2G_expr(schedule, unpair(grid_i), unpair(particles_p), unpair(mpvalues_ip), partition, split_equations(equations))
end

function P2G_expr(schedule::QuoteNode, (grid,i), (particles,p), (mpvalues,ip), partition, equations::Vector)
    issum = map(eq -> eq.issumeq, equations)
    (!allequal(issum) && issorted(issum)) && error("@P2G: Equations without `@∑` must come after those with `@∑`")

    code = quote
        Tesserae.check_arguments_for_P2G($grid, $particles, $mpvalues, $partition)
    end

    sum_equations = equations[issum]
    if !isempty(sum_equations)
        pre, body = P2G_sum_expr((grid,i), (particles,p), (mpvalues,ip), sum_equations)
        if !DEBUG
            body = :(@inbounds $body)
        end
        code = quote
            $code
            $pre
            Tesserae.P2G(($grid, $particles, $mpvalues, $p) -> $body, Tesserae.get_device($grid), Val($schedule), $grid, $particles, $mpvalues, $partition)
        end
    end

    nosum_equations = equations[.!issum]
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

function P2G_sum_expr((grid,i), (particles,p), (mpvalues,ip), sum_equations::Vector)
    @gensym mp gridindices

    maps = [grid=>i, particles=>p, mp=>ip]
    replaced = [Set{Expr}(), Set{Expr}(), Set{Expr}()]
    for k in eachindex(sum_equations)
        eq = sum_equations[k]
        @assert Meta.isexpr(eq.lhs, :ref)
        eq.lhs = resolve_refs(eq.lhs, maps)
        eq.rhs = resolve_refs(eq.rhs, maps; replaced)
    end

    fillzeros = Any[]
    for k in eachindex(sum_equations)
        (; lhs, rhs, op) = sum_equations[k]
        op == :(=)  && push!(fillzeros, :(Tesserae.fillzero!($(remove_indexing(lhs)))))
        op == :(-=) && (rhs = :(-$rhs))
        sum_equations[k] = :(Tesserae.add!($(lhs.args...), $rhs))
    end

    body = quote
        $(replaced[2]...)
        $mp = $mpvalues[$p]
        $gridindices = neighboringnodes($mp, $grid)
        for $ip in eachindex($gridindices)
            $i = $gridindices[$ip]
            $(union(replaced[1], replaced[3])...)
            $(sum_equations...)
        end
    end
    
    Expr(:block, fillzeros...), body
end

# CPU: sequential
function P2G(f, ::CPUDevice, ::Val{scheduler}, grid, particles, mpvalues, ::Nothing) where {scheduler}
    scheduler == :nothing || @warn "@P2G: `ColorPartition` must be given for threaded computation" maxlog=1
    for p in eachindex(particles)
        @inline f(grid, particles, mpvalues, p)
    end
end

# CPU: multi-threading
function P2G(f, ::CPUDevice, ::Val{scheduler}, grid, particles, mpvalues, partition::ColorPartition) where {scheduler}
    strat = strategy(partition)
    for group in colorgroups(strat)
        tforeach(group, scheduler) do index
            @_inline_meta
            for p in particle_indices_in(strat, index)
                @inline f(grid, particles, mpvalues, p)
            end
        end
    end
end

# GPU
@kernel function gpukernel_P2G(f, grid, @Const(particles), @Const(mpvalues))
    p = @index(Global)
    f(grid, particles, mpvalues, p)
end
function P2G(f, device::GPUDevice, ::Val{scheduler}, grid, particles, mpvalues, ::Nothing) where {scheduler}
    scheduler == :nothing || @warn "Multi-threading is disabled for GPU" maxlog=1
    backend = get_backend(device)
    kernel = gpukernel_P2G(backend, 256)
    kernel(f, hybrid(grid), particles, mpvalues; ndrange=length(particles))
    synchronize(backend)
end

function P2G_nosum_expr((grid,i), nosum_equations::Vector)
    maps = [grid=>i]
    for k in eachindex(nosum_equations)
        eq = nosum_equations[k]
        eq.lhs = remove_indexing(resolve_refs(eq.lhs, maps))
        eq.rhs = remove_indexing(resolve_refs(eq.rhs, maps))
        nosum_equations[k] = eq
    end

    map!(nosum_equations, nosum_equations) do eq
        (; lhs, rhs, op) = eq
        if @capture(lhs, $grid.x_)
            :(@. $(Expr(op, lhs, rhs)))
        else # TODO: avoid intermediate allocation
            Expr(op, lhs, :(@. $rhs))
        end
    end
    Expr(:block, nosum_equations...)
end

function check_arguments_for_P2G(grid, particles, mpvalues, partition)
    get_mesh(grid) isa AbstractMesh || error("@P2G: grid must have a mesh")
    eltype(mpvalues) <: MPValue || error("@P2G: invalid `MPValue`s, got type $(typeof(mpvalues))")
    if grid isa SpGrid
        if length(propertynames(grid)) > 1
            isempty(get_data(getproperty(grid, 2))) && error("@P2G: SpGrid indices not activated")
        end
    end
    @assert length(particles) ≤ length(mpvalues)
    if partition isa ColorPartition
        strat = strategy(partition)
        if strat isa BlockStrategy
            @assert blocksize(grid) == blocksize(strat)
            sum(length(particle_indices_in(strat, blk)) for blk in blockindices(strat)) == 0 &&
                error("@P2G: No particles assigned to any block in ColorPartition")
        end
    end
    # check device
    device = get_device(grid)
    @assert get_device(particles) == get_device(mpvalues) == device
end

"""
    @G2P grid=>i particles=>p mpvalues=>ip begin
        equations...
    end

Grid-to-particle transfer macro.
Based on the `parent => index` expressions, `a[index]` in `equations`
translates to `parent.a[index]`. This `index` can be replaced with
any other name.

# Examples
```julia
@G2P grid=>i particles=>p mpvalues=>ip begin

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
    mp = mpvalues[p]
    nodeindices = neighboringnodes(mp)
    Δvₚ = zero(eltype(particles.v))
    ∇vₚ = zero(eltype(particles.∇v))
    Δxₚ = zero(eltype(particles.x))
    for ip in eachindex(nodeindices)
        i = nodeindices[ip]
        Δvₚ += mp.w[ip] * (grid.vⁿ[i] - grid.v[i])
        ∇vₚ += grid.v[i] ⊗ mp.∇w[ip]
        Δxₚ += mp.w[ip] * grid.v[i] * Δt
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
macro G2P(grid_i, particles_p, mpvalues_ip, equations)
    G2P_expr(QuoteNode(:nothing), grid_i, particles_p, mpvalues_ip, equations)
end
macro G2P(schedule::QuoteNode, grid_i, particles_p, mpvalues_ip, equations)
    G2P_expr(schedule, grid_i, particles_p, mpvalues_ip, equations)
end

function G2P_expr(schedule::QuoteNode, grid_i::Expr, particles_p::Expr, mpvalues_ip::Expr, equations::Expr)
    G2P_expr(schedule, unpair(grid_i), unpair(particles_p), unpair(mpvalues_ip), split_equations(equations))
end

function G2P_expr(schedule::QuoteNode, (grid,i), (particles,p), (mpvalues,ip), equations::Vector)
    issum = map(eq -> eq.issumeq, equations)
    (!allequal(issum) && issorted(issum)) && error("@P2G: Equations without `@∑` must come after those with `@∑`")

    code = quote
        Tesserae.check_arguments_for_G2P($grid, $particles, $mpvalues)
    end

    if !isempty(equations)
        body = G2P_sum_expr((grid,i), (particles,p), (mpvalues,ip), equations[issum], equations[.!issum])
        if !DEBUG
            body = :(@inbounds $body)
        end
        code = quote
            $code
            Tesserae.G2P(($grid, $particles, $mpvalues, $p) -> $body, Tesserae.get_device($grid), Val($schedule), $grid, $particles, $mpvalues)
        end
    end

    esc(prettify(code; lines=true, alias=false))
end

# CPU: sequential & multi-threading
function G2P(f, ::CPUDevice, ::Val{scheduler}, grid, particles, mpvalues) where {scheduler}
    tforeach(eachindex(particles), scheduler) do p
        @inline f(grid, particles, mpvalues, p)
    end
end

# GPU
@kernel function gpukernel_G2P(f, @Const(grid), particles, @Const(mpvalues))
    p = @index(Global)
    f(grid, particles, mpvalues, p)
end
function G2P(f, device::GPUDevice, ::Val{scheduler}, grid, particles, mpvalues) where {scheduler}
    scheduler == :nothing || @warn "Multi-threading is disabled for GPU" maxlog=1
    backend = get_backend(device)
    kernel = gpukernel_G2P(backend, 256)
    kernel(f, grid, particles, mpvalues; ndrange=length(particles))
    synchronize(backend)
end

function G2P_sum_expr((grid,i), (particles,p), (mpvalues,ip), sum_equations::Vector, nosum_equations::Vector)
    @gensym mp gridindices

    code = Expr(:block)

    if !isempty(sum_equations)
        maps = [grid=>i, particles=>p, mp=>ip]
        replaced = [Set{Expr}(), Set{Expr}(), Set{Expr}()]
        for k in eachindex(sum_equations)
            eq = sum_equations[k]
            @assert Meta.isexpr(eq.lhs, :ref)
            eq.lhs = resolve_refs(eq.lhs, maps)
            eq.rhs = resolve_refs(eq.rhs, maps; replaced)
        end

        inits = []
        saves = []
        for k in eachindex(sum_equations)
            (; lhs, rhs, op) = sum_equations[k]
            tmp = Symbol(lhs, :_p)
            push!(inits, :($tmp = zero(eltype($(remove_indexing(lhs))))))
            push!(saves, Expr(op, lhs, tmp))
            sum_equations[k] = :($tmp += $rhs)
        end

        code = quote
            $(replaced[2]...)
            $(inits...)
            $mp = $mpvalues[$p]
            $gridindices = neighboringnodes($mp, $grid)
            for $ip in eachindex($gridindices)
                $i = $gridindices[$ip]
                $(union(replaced[1], replaced[3])...)
                $(sum_equations...)
            end
            $(saves...)
        end
    end

    if !isempty(nosum_equations)
        for k in eachindex(nosum_equations)
            eq = nosum_equations[k]
            eq.lhs = resolve_refs(eq.lhs, maps)
            eq.rhs = resolve_refs(eq.rhs, maps)
            nosum_equations[k] = Expr(eq.op, eq.lhs, eq.rhs)
        end
        code = quote
            $code
            $(nosum_equations...)
        end
    end

    code
end

"""
    @G2P2G grid=>i particles=>p mpvalues=>ip [partition] begin
        equations...
    end

Combined grid-to-particle and particle-to-grid transfer macro.

Allows both [`@G2P`](@ref) (interpolation from grid to particles) and [`@P2G`](@ref) (scattering from particles to grid)
to be performed in a single loop over particles, avoiding repeated traversals.

# Examples
```julia
@G2P2G grid=>i particles=>p mpvalues=>ip begin
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
macro G2P2G(grid_i, particles_p, mpvalues_ip, equations)
    G2P2G_expr(QuoteNode(:nothing), grid_i, particles_p, mpvalues_ip, nothing, equations)
end
macro G2P2G(grid_i, particles_p, mpvalues_ip, partition, equations)
    G2P2G_expr(QuoteNode(:nothing), grid_i, particles_p, mpvalues_ip, partition, equations)
end
macro G2P2G(schedule::QuoteNode, grid_i, particles_p, mpvalues_ip, equations)
    G2P2G_expr(schedule, grid_i, particles_p, mpvalues_ip, nothing, equations)
end
macro G2P2G(schedule::QuoteNode, grid_i, particles_p, mpvalues_ip, partition, equations)
    G2P2G_expr(schedule, grid_i, particles_p, mpvalues_ip, partition, equations)
end

function G2P2G_expr(schedule::QuoteNode, grid_i::Expr, particles_p::Expr, mpvalues_ip::Expr, partition, equations::Expr)
    G2P2G_expr(schedule, unpair(grid_i), unpair(particles_p), unpair(mpvalues_ip), partition, split_equations(equations))
end

function G2P2G_expr(schedule::QuoteNode, (grid,i), (particles,p), (mpvalues,ip), partition, equations::Vector)
    equations_g2p_sum = Any[]
    equations_p2g_sum = Any[]
    equations_g2p_nosum = Any[]
    equations_p2g_nosum = Any[]
    precedence = 1
    for k in eachindex(equations)
        eq = equations[k]
        if eq.issumeq
            @assert @capture(eq.lhs, A_[index_])
            if index == p
                @assert precedence == 1
                push!(equations_g2p_sum, eq)
            elseif index == i
                @assert precedence ≤ 3
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

    code = quote
        Tesserae.check_arguments_for_G2P($grid, $particles, $mpvalues)
        Tesserae.check_arguments_for_P2G($grid, $particles, $mpvalues, $partition)
    end
    body = Expr(:block)

    if !isempty(equations_g2p_sum) || !isempty(equations_g2p_nosum)
        expr = G2P_sum_expr((grid,i), (particles,p), (mpvalues,ip), equations_g2p_sum, equations_g2p_nosum)
        body = quote
            $body
            $expr
        end
    end

    if !isempty(equations_p2g_sum)
        pre, expr = P2G_sum_expr((grid,i), (particles,p), (mpvalues,ip), equations_p2g_sum)
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
        Tesserae.P2G(($grid, $particles, $mpvalues, $p) -> $body, Tesserae.get_device($grid), Val($schedule), $grid, $particles, $mpvalues, $partition)
    end

    if !isempty(equations_p2g_nosum)
        body = P2G_nosum_expr((grid,i), equations_p2g_nosum)
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
    @assert @capture(ex, lhs_ => rhs_)
    lhs::Symbol, rhs::Symbol
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

function split_equations(expr::Expr)::Vector{Any}
    expr = MacroTools.prewalk(MacroTools.rmlines, expr)
    @assert @capture(expr, begin exprs__ end)
    map(exprs) do ex
        dict = MacroTools.trymatch(Expr(:op_, :lhs_, :rhs_), ex)
        dict === nothing && error("wrong expression: $ex")
        lhs, rhs, op = dict[:lhs], dict[:rhs], dict[:op]
        if @capture(rhs, @∑ eq_)
            (op == :(=) || op == :(+=) || op == :(-=)) || error("@∑ is only allowed on the RHS of assignments with `=`, `+=`, or `-=`, got $ex")
            return Equation(true, lhs, eq, op)
        end
        has_sum_macro(rhs) && error("@∑ must appear alone as the entire RHS expression, got $ex")
        Equation(false, lhs, rhs, op)
    end
end

function resolve_refs(expr, maps::Vector{Pair{Symbol, Symbol}}; replaced::Union{Nothing, Vector{Set{Expr}}} = nothing) # maps: [:grid=>:i, :particles=>:p, ...]
    replaced === nothing || @assert length(maps) == length(replaced)
    MacroTools.postwalk(expr) do ex
        if @capture(ex, x_[i_])
            for (k, (parent, j)) in enumerate(maps)
                if i == j # same index
                    resolved = :($parent.$x[$i])
                    replaced === nothing && return resolved
                    sym = Symbol(resolved)
                    push!(replaced[k], :($sym = $resolved))
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

function check_arguments_for_G2P(grid, particles, mpvalues)
    get_mesh(grid) isa AbstractMesh || error("@G2P: grid must have a mesh")
    eltype(mpvalues) <: MPValue || error("@G2P: invalid `MPValue`s, got type $(typeof(mpvalues))")
    if grid isa SpGrid
        if length(propertynames(grid)) > 1
            isempty(get_data(getproperty(grid, 2))) && error("@G2P: SpGrid indices not activated")
        end
    end
    @assert length(particles) ≤ length(mpvalues)
    # check device
    device = get_device(grid)
    @assert get_device(particles) == get_device(mpvalues) == device
end
