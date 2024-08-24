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
    v[i]  = vⁿ[i] + Δt * (f[i] / m[i])

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
@. grid.v  = grid.vⁿ + Δt * (grid.f / grid.m)
```

!!! warning
    In `@P2G`, `Calculation on grid` part must be placed after
    `Particle-to-grid transfer` part.
"""
macro P2G(grid_pair, particles_pair, mpvalues_pair, equations)
    P2G_macro(QuoteNode(:nothing), grid_pair, particles_pair, mpvalues_pair, nothing, equations)
end
macro P2G(grid_pair, particles_pair, mpvalues_pair, space, equations)
    P2G_macro(QuoteNode(:nothing), grid_pair, particles_pair, mpvalues_pair, space, equations)
end
macro P2G(schedule::QuoteNode, grid_pair, particles_pair, mpvalues_pair, equations)
    P2G_macro(schedule, grid_pair, particles_pair, mpvalues_pair, nothing, equations)
end
macro P2G(schedule::QuoteNode, grid_pair, particles_pair, mpvalues_pair, space, equations)
    P2G_macro(schedule, grid_pair, particles_pair, mpvalues_pair, space, equations)
end

function P2G_macro(schedule::QuoteNode, grid_pair, particles_pair, mpvalues_pair, space, equations)
    grid, i = unpair(grid_pair)
    particles, _ = unpair(particles_pair)
    mpvalues, _ = unpair(mpvalues_pair)

    @assert equations.head == :block
    Base.remove_linenums!(equations)
    replace_dollar_by_identity!(equations)
    sumornot = map(ex->issumexpr(ex, i), equations.args)
    if sort(sumornot; rev=true) != sumornot
        error("@P2G: Equations without `@∑` must come after those with `@∑`")
    end

    sum_equations = equations.args[sumornot]
    nosum_equations = equations.args[.!sumornot]

    body1 = P2G_sum_macro(schedule, grid_pair, particles_pair, mpvalues_pair, space, sum_equations)
    body2 = P2G_nosum_macro(schedule, grid_pair, nosum_equations)

    quote
        $check_arguments_for_P2G($grid, $particles, $mpvalues, $space)
        $body1
        $body2
    end |> esc
end

function P2G_sum_macro(schedule::QuoteNode, grid_pair, particles_pair, mpvalues_pair, space, sum_equations::Vector)
    isempty(sum_equations) && return Expr(:block)

    grid, i = unpair(grid_pair)
    particles, p = unpair(particles_pair)
    mpvalues, ip = unpair(mpvalues_pair)
    @gensym mp gridindices

    pairs = [grid=>i, particles=>p, mp=>ip]
    vars = [Set{Expr}(), Set{Expr}(), Set{Expr}()]
    foreach(ex->complete_sumeq_expr!(ex, pairs, vars), sum_equations)

    sum_names = Any[]
    init_gridprops = Any[]
    for ex in sum_equations
        # `lhs` is, for example, `grid.m[i]`
        lhs = ex.args[1]
        if Meta.isexpr(ex, :(=))
            push!(init_gridprops, :(Tesserae.fillzero!($(lhs.args[1]))))
        end
        push!(sum_names, lhs.args[1].args[2].value) # extract `m` for `grid.m[i]`
    end

    foreach(ex->ex.head=:(+=), sum_equations)

    body = quote
        $(vars[2]...)
        $mp = $mpvalues[$p]
        $gridindices = neighboringnodes($mp, $grid)
        for $ip in eachindex($gridindices)
            $i = $gridindices[$ip]
            $(union(vars[1], vars[3])...)
            $(sum_equations...)
        end
    end

    if !DEBUG
        body = :(@inbounds $body)
    end
    
    quote
        $(init_gridprops...)
        $(P2G_expr(schedule, grid, particles, mpvalues, space, p, body))
    end
end

function P2G_expr(schedule::QuoteNode, grid, particles, mpvalues, space, p, body)
    if isnothing(space)
        body = quote
            $(schedule.value != :nothing) && @warn "@P2G: `BlockSpace` must be given for threaded computation" maxlog=1
            for $p in Tesserae.eachparticleindex($particles, $mpvalues)
                $body
            end
        end
    else
        @gensym blocks blk
        body = :(for $blk in $blocks
                     for $p in $space[$blk]
                         $body
                     end
                 end)
        if schedule.value != :nothing
            body = :(@threaded $schedule $body)
        end
        body = quote
            for $blocks in Tesserae.threadsafe_blocks($space)
                $body
            end
        end
    end
    body
end

function P2G_nosum_macro(schedule, grid_pair, nosum_equations::Vector)
    isempty(nosum_equations) && return Expr(:block)

    grid, i = unpair(grid_pair)
    foreach(ex->complete_parent_from_index!(ex, [grid=>i]), nosum_equations)

    vars = Set{Expr}()
    foreach(ex->findarrays_from_index!(vars, i, ex), nosum_equations)

    body = quote
        Tesserae.foreach_gridindex(Val($schedule), Tesserae.GridIndexStyle($(vars...)), $grid) do $i
            Base.@_inline_meta
            $(nosum_equations...)
        end
    end

    body
end

struct IndexSpArray end
GridIndexStyle(::Type{T}) where {T <: AbstractArray} = IndexStyle(T)
GridIndexStyle(::Type{<: SpArray}) = IndexSpArray()
GridIndexStyle(A::AbstractArray) = GridIndexStyle(typeof(A))
GridIndexStyle(A::AbstractArray, B::AbstractArray) = GridIndexStyle(GridIndexStyle(A), GridIndexStyle(B))
GridIndexStyle(A::AbstractArray, B::AbstractArray...) = GridIndexStyle(GridIndexStyle(A), GridIndexStyle(B...))
GridIndexStyle(::IndexLinear, ::IndexLinear) = IndexLinear()
GridIndexStyle(::IndexSpArray, ::IndexSpArray) = IndexSpArray()
GridIndexStyle(::IndexStyle, ::IndexStyle) = IndexCartesian()

for schedule in QuoteNode.((:nothing, :static, :dynamic))
    wrap(ex, sch) = sch.value==:nothing ? :(@simd $ex) : :(@threaded $sch $ex)

    body = wrap(:(for i in eachindex(style, grid)
                      @inbounds f(i)
                  end), schedule)
    @eval foreach_gridindex(f, ::Val{$schedule}, style::IndexStyle, grid::Grid) = $body

    body = wrap(:(for i in eachindex(style, grid)
                      @inbounds if isactive(grid, i)
                          f(i)
                      end
                  end), schedule)
    @eval foreach_gridindex(f, ::Val{$schedule}, style::IndexCartesian, grid::SpGrid) = $body

    body = wrap(:(for i in eachindex(style, grid)
                      @inbounds f(UnsafeSpIndex(i))
                  end), schedule)
    @eval foreach_gridindex(f, ::Val{$schedule}, style::IndexSpArray, grid::SpGrid) = $body
end

struct UnsafeSpIndex{I}
    i::I
end
@inline Base.getindex(A::SpArray, i::UnsafeSpIndex) = (@_propagate_inbounds_meta; get_data(A)[i.i])
@inline Base.setindex!(A::SpArray, v, i::UnsafeSpIndex) = (@_propagate_inbounds_meta; get_data(A)[i.i]=v; A)

function check_arguments_for_P2G(grid, particles, mpvalues, space)
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
    v[p] += @∑ (vⁿ[i] - v[i]) * w[ip]
    ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
    x[p] += @∑ Δt * v[i] * w[ip]

    # Calculation on particle
    Δϵₚ = Δt * symmetric(∇v[p])
    F[p]  = (I + Δt*∇v[p]) ⋅ F[p]
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
        Δvₚ += (grid.vⁿ[i] - grid.v[i]) * mp.w[ip]
        ∇vₚ += grid.v[i] ⊗ mp.∇w[ip]
        Δxₚ += Δt * grid.v[i] * mp.w[ip]
    end
    particles.v[p] += Δvₚ
    particles.∇v[p] = ∇vₚ
    particles.x[p] += Δxₚ
end

# Calculation on particle
for p in eachindex(particles)
    Δϵₚ = Δt * symmetric(particles.∇v[p])
    particles.F[p]  = (I + Δt*particles.∇v[p]) ⋅ particles.F[p]
    particles.V[p]  = particles.V⁰[p] * det(particles.F[p])
    particles.σ[p] += λ*tr(Δϵₚ)*I + 2μ*Δϵₚ # Linear elastic material
end
```

!!! warning
    In `@G2P`, `Calculation on particles` part must be placed after
    `Grid-to-particle transfer` part.
"""
macro G2P(grid_pair, particles_pair, mpvalues_pair, equations)
    G2P_macro(QuoteNode(:nothing), grid_pair, particles_pair, mpvalues_pair, equations)
end
macro G2P(schedule::QuoteNode, grid_pair, particles_pair, mpvalues_pair, equations)
    G2P_macro(schedule, grid_pair, particles_pair, mpvalues_pair, equations)
end

function G2P_macro(schedule::QuoteNode, grid_pair, particles_pair, mpvalues_pair, equations)
    grid, _ = unpair(grid_pair)
    particles, p = unpair(particles_pair)
    mpvalues, _ = unpair(mpvalues_pair)

    @assert equations.head == :block
    Base.remove_linenums!(equations)
    replace_dollar_by_identity!(equations)
    sumornot = map(ex->issumexpr(ex, p), equations.args)
    if sort(sumornot; rev=true) != sumornot
        error("@P2G: Equations without `@∑` must come after those with `@∑`")
    end

    sum_equations = equations.args[sumornot]
    nosum_equations = equations.args[.!sumornot]

    body1 = G2P_sum_macro(grid_pair, particles_pair, mpvalues_pair, sum_equations)
    body2 = G2P_nosum_macro(particles_pair, nosum_equations)

    body = quote
        $check_arguments_for_G2P($grid, $particles, $mpvalues)
        $body1
        $body2
    end

    if !DEBUG
        body = :(@inbounds $body)
    end

    if isallunderscore(mpvalues)
        iterator = :(eachindex($particles))
    else
        iterator = :(Tesserae.eachparticleindex($particles, $mpvalues))
    end

    if schedule.value == :nothing
        body = quote
            for $p in $iterator
                $body
            end
        end
    else
        body = quote
            @threaded $schedule for $p in $iterator
                $body
            end
        end
    end

    esc(body)
end

function G2P_sum_macro(grid_pair, particles_pair, mpvalues_pair, sum_equations::Vector)
    isempty(sum_equations) && return Expr(:block)

    grid, i = unpair(grid_pair)
    particles, p = unpair(particles_pair)
    mpvalues, ip = unpair(mpvalues_pair)
    @gensym mp gridindices

    pairs = [grid=>i, particles=>p, mp=>ip]
    vars = [Set{Expr}(), Set{Expr}(), Set{Expr}()]
    foreach(ex->complete_sumeq_expr!(ex, pairs, vars), sum_equations)

    particles_vars_declare = []
    particles_vars_store = []
    for ex in sum_equations
        lhs = ex.args[1]
        name_p = Symbol(lhs.args[1], :_p)
        push!(particles_vars_declare, :($name_p = zero(eltype($(lhs.args[1])))))
        push!(particles_vars_store, Expr(ex.head, ex.args[1], name_p))
        ex.args[1] = name_p
    end

    foreach(ex->ex.head=:(+=), sum_equations)

    quote
        $(vars[2]...)
        $(particles_vars_declare...)
        $mp = $mpvalues[$p]
        $gridindices = neighboringnodes($mp, $grid)
        for $ip in eachindex($gridindices)
            $i = $gridindices[$ip]
            $(union(vars[1], vars[3])...)
            $(sum_equations...)
        end
        $(particles_vars_store...)
    end
end

function G2P_nosum_macro(particles_pair, nosum_equations::Vector)
    isempty(nosum_equations) && return Expr(:block)

    particles, p = unpair(particles_pair)
    foreach(ex->complete_parent_from_index!(ex, [particles=>p]), nosum_equations)

    Expr(:block, nosum_equations...)
end

function unpair(expr::Expr)
    @assert expr.head==:call && expr.args[1]==:(=>) && isa(expr.args[2],Symbol) && isa(expr.args[3],Symbol)
    expr.args[2], expr.args[3]
end
function unpair(s::Symbol)
    @assert isallunderscore(s)
    s, s
end

function issumexpr(expr::Expr, inds::Symbol...)
    if length(expr.args) == 2 && _issumexpr(expr.args[2])
        @assert (expr.head==:(=) || expr.head==:(+=) || expr.head==:(-=)) && isrefexpr(expr.args[1], inds...)
    else
        false
    end
    _issumexpr(expr.args[2])
end

function _issumexpr(expr::Expr)
    if expr.head==:macrocall
        @assert length(expr.args)==3 && (expr.args[1]==Symbol("@∑") || expr.args[1]==Symbol("@Σ")) && isa(expr.args[2],LineNumberNode)
        true
    else
        false
    end
end
_issumexpr(x) = false

isrefexpr(expr::Expr, inds::Symbol...) = expr.head==:ref && all(expr.args[2:end] .== inds)
isrefexpr(x, inds...) = false

function complete_sumeq_expr!(expr::Expr, pairs::Vector{Pair{Symbol, Symbol}}, vars::Vector)
    # must check `iseqexpr` in advance
    expr.args[2] = remove_∑(expr.args[2])
    complete_parent_from_index!(expr.args[1], pairs)
    complete_sumeq_rhs_expr!(Meta.quot(expr.args[2]), pairs, vars) # rhs
end

function remove_∑(rhs::Expr)
    rhs.args[3] # extract only inside of @∑ (i.e., remove @∑)
end

function complete_parent_from_index!(expr::Expr, pairs::Vector{Pair{Symbol, Symbol}})
    if Meta.isexpr(expr, :ref) && length(expr.args) == 2 # support only single index
        for p in pairs
            if p.second == expr.args[2] # same index
                expr.args[1] = :($(p.first).$(expr.args[1]))
            end
        end
    end
    for ex in expr.args
        complete_parent_from_index!(ex, pairs)
    end
end
complete_parent_from_index!(expr, pairs::Vector{Pair{Symbol, Symbol}}) = nothing

function complete_sumeq_rhs_expr!(expr::Expr, pairs::Vector{Pair{Symbol, Symbol}}, vars::Vector)
    for i in eachindex(expr.args)
        ex = expr.args[i]
        if Meta.isexpr(ex, :ref) && length(ex.args) == 2 # support only single index
            index = ex.args[2]
            for j in eachindex(pairs)
                if pairs[j].second == index
                    name = Symbol(ex)
                    push!(vars[j], :($name = $(pairs[j].first).$(ex.args[1])[$(ex.args[2])]))
                    expr.args[i] = name
                end
            end
        end
        # check for recursive indexing
        complete_sumeq_rhs_expr!(ex, pairs, vars)
    end
end
complete_sumeq_rhs_expr!(expr, pairs::Vector{Pair{Symbol, Symbol}}, vars::Vector) = nothing

function findarrays_from_index!(set::Set{Expr}, index::Symbol, expr::Expr)
    if Meta.isexpr(expr, :ref) && length(expr.args)==2 && expr.args[2]==index
        push!(set, expr.args[1])
    end
    for ex in expr.args
        findarrays_from_index!(set, index, ex)
    end
end
findarrays_from_index!(set::Set{Expr}, index::Symbol, expr) = nothing

function eachparticleindex(particles::AbstractVector, mpvalues::AbstractVector{<: MPValue})
    @assert length(particles) ≤ length(mpvalues)
    eachindex(particles)
end

isallunderscore(s::Symbol) = all(==('_'), string(s))

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
    if grid isa SpGrid
        if length(propertynames(grid)) > 1
            isempty(get_data(getproperty(grid, 2))) && error("@G2P: SpGrid indices not activated")
        end
    end
    @assert length(particles) ≤ length(mpvalues)
end
