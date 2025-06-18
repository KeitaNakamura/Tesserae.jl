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

@inline add!(A::HybridArray{T}, i, v::T) where {T} = (@_propagate_inbounds_meta; _add!(get_device(A), A, i, v))
@inline _add!(::CPUDevice, A::HybridArray, i, v) = (@_propagate_inbounds_meta; A[i] += v)
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
    @P2G grid=>i particles=>p mpvalues=>ip [space] begin
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
    f[i]  = @∑ -V[p] * σ[p] ⋅ ∇w[ip]

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
        grid.mv[i] += -particles.V[p] * particles.σ[p] ⋅ mp.∇w[ip]
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
macro P2G(grid_i, particles_p, mpvalues_ip, space, equations)
    P2G_expr(QuoteNode(:nothing), grid_i, particles_p, mpvalues_ip, space, equations)
end
macro P2G(schedule::QuoteNode, grid_i, particles_p, mpvalues_ip, equations)
    P2G_expr(schedule, grid_i, particles_p, mpvalues_ip, nothing, equations)
end
macro P2G(schedule::QuoteNode, grid_i, particles_p, mpvalues_ip, space, equations)
    P2G_expr(schedule, grid_i, particles_p, mpvalues_ip, space, equations)
end

function P2G_expr(schedule::QuoteNode, grid_i::Expr, particles_p::Expr, mpvalues_ip::Expr, space, equations::Expr)
    P2G_expr(schedule, unpair(grid_i), unpair(particles_p), unpair(mpvalues_ip), space, split_equations(equations))
end

function P2G_expr(schedule::QuoteNode, (grid,i), (particles,p), (mpvalues,mp), space, equations::Vector)
    issum = map(eq -> eq.issumeq, equations)
    (!allequal(issum) && issorted(issum)) && error("@P2G: Equations without `@∑` must come after those with `@∑`")

    body1 = P2G_sum_expr(schedule, (grid,i), (particles,p), (mpvalues,mp), space, equations[issum])
    body2 = P2G_nosum_expr(schedule, (grid,i), equations[.!issum])

    quote
        $check_arguments_for_P2G($grid, $particles, $mpvalues, $space)
        $body1
        $body2
    end |> esc
end

function P2G_sum_expr(schedule::QuoteNode, (grid,i), (particles,p), (mpvalues,ip), space, sum_equations::Vector)
    isempty(sum_equations) && return Expr(:block)

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
        sum_equations[k] = :($add!($(lhs.args...), $rhs))
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

    if !DEBUG
        body = :(@inbounds $body)
    end
    
    quote
        $(fillzeros...)
        $P2G(($grid, $particles, $mpvalues, $p) -> $body, $get_device($grid), Val($schedule), $hybrid($grid), $particles, $mpvalues, $space)
    end
end

# CPU: sequential
function P2G(f, ::CPUDevice, ::Val{scheduler}, grid, particles, mpvalues, ::Nothing) where {scheduler}
    scheduler == :nothing || @warn "@P2G: `BlockSpace` must be given for threaded computation" maxlog=1
    for p in eachindex(particles)
        @inline f(grid, particles, mpvalues, p)
    end
end

# CPU: multi-threading
function P2G(f, ::CPUDevice, ::Val{scheduler}, grid, particles, mpvalues, space::BlockSpace) where {scheduler}
    for blocks in threadsafe_blocks(space)
        tforeach(blocks, scheduler) do blk
            @_inline_meta
            for p in space[blk]
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
    kernel(f, grid, particles, mpvalues; ndrange=length(particles))
    synchronize(backend)
end

function P2G_nosum_expr(schedule, (grid,i), nosum_equations::Vector)
    isempty(nosum_equations) && return Expr(:block)

    maps = [grid=>i]
    for k in eachindex(nosum_equations)
        eq = nosum_equations[k]
        eq.lhs = remove_indexing(resolve_refs(eq.lhs, maps))
        eq.rhs = remove_indexing(resolve_refs(eq.rhs, maps))
        nosum_equations[k] = eq
    end

    map!(nosum_equations, nosum_equations) do eq
        ex = Expr(eq.op, eq.lhs, eq.rhs)
        :(@. $ex)
    end
    Expr(:block,nosum_equations...)
end

function check_arguments_for_P2G(grid, particles, mpvalues, space)
    get_mesh(grid) isa AbstractMesh || error("@P2G: grid must have a mesh")
    eltype(mpvalues) <: MPValue || error("@P2G: invalid `MPValue`s, got type $(typeof(mpvalues))")
    if grid isa SpGrid
        if length(propertynames(grid)) > 1
            isempty(get_data(getproperty(grid, 2))) && error("@P2G: SpGrid indices not activated")
        end
    end
    @assert length(particles) ≤ length(mpvalues)
    if space isa BlockSpace
        @assert blocksize(grid) == size(space)
        sum(length, space) == 0 && error("@P2G: BlockSpace not activated")
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
    F[p]  = (I + ∇v[p]*Δt) ⋅ F[p]
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
    particles.F[p]  = (I + particles.∇v[p]*Δt) ⋅ particles.F[p]
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

    body = G2P_sum_expr((grid,i), (particles,p), (mpvalues,ip), equations[issum], equations[.!issum])

    if !DEBUG
        body = :(@inbounds $body)
    end

    body = quote
        $check_arguments_for_G2P($grid, $particles, $mpvalues)
        $G2P(($grid, $particles, $mpvalues, $p) -> $body, $get_device($grid), Val($schedule), $grid, $particles, $mpvalues)
    end

    esc(body)
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
    isempty(sum_equations) && return Expr(:block)

    @gensym mp gridindices

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

    for k in eachindex(nosum_equations)
        eq = nosum_equations[k]
        eq.lhs = resolve_refs(eq.lhs, maps)
        eq.rhs = resolve_refs(eq.rhs, maps)
        nosum_equations[k] = Expr(eq.op, eq.lhs, eq.rhs)
    end

    quote
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
        $(nosum_equations...)
    end
end

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
    replace_dollar_by_identity!(expr)
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

function replace_dollar_by_identity!(expr::Expr)
    if Meta.isexpr(expr, :$)
        expr.head = :call
        pushfirst!(expr.args, :identity)
    end
    for ex in expr.args
        replace_dollar_by_identity!(ex)
    end
    expr
end
replace_dollar_by_identity!(x) = x

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
