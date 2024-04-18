"""
    @P2G grid=>i particles=>p mpvalues=>ip [space] begin
        equations...
    end

Particle-to-grid transfer macro.

# Examples
```jl
@P2G grid=>i particles=>p mpvalues=>ip begin

    # particle-to-grid transfer
    m[i] = @∑ N[ip] * m[p]
    mv[i] = @∑ N[ip] * m[p] * (v[p] + ∇v[p] ⋅ (x[i] - x[p]))
    f[i] = @∑ -V[p] * σ[p] ⋅ ∇N[ip]

    # calculation on grid
    vⁿ[i] = mv[i] / m[i]
    v[i] = vⁿ[i] + Δt * (f[i] / m[i])

end
```
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
    _, i = unpair(grid_pair)

    @assert equations.head == :block
    Base.remove_linenums!(equations)
    sumornot = map(ex->issumexpr(ex, i), equations.args)
    if sort(sumornot; rev=true) != sumornot
        error("@P2G: Equations without `@∑` must come after those with `@∑`")
    end

    sum_equations = equations.args[sumornot]
    nosum_equations = equations.args[.!sumornot]

    body1 = P2G_sum_macro(schedule, grid_pair, particles_pair, mpvalues_pair, space, sum_equations)
    body2 = P2G_nosum_macro(schedule, grid_pair, nosum_equations)

    quote
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
            push!(init_gridprops, :(Sequoia.fillzero!($(lhs.args[1]))))
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
        $pre_P2G($grid, $particles, $mpvalues, $space, $(Val(tuple(sum_names...))))
        $P2G(Val($schedule), $grid, $particles, $mpvalues, $space) do $p, $grid, $particles, $mpvalues
            Base.@_inline_meta
            $body
        end
        $post_P2G($grid, $particles, $mpvalues, $space, $(Val(tuple(sum_names...))))
    end
end

for schedule in QuoteNode.((:nothing, :static, :dynamic))

    # simple for loop
    @eval function P2G(f, ::Val{$schedule}, grid::Grid, particles::AbstractVector, mpvalues::AbstractVector{<: MPValue}, ::Nothing)
        $schedule != :nothing && @warn "@P2G: `BlockSpace` must be given for threaded computation" maxlog=1
        for p in Sequoia.eachparticleindex(particles, mpvalues)
            f(p, grid, particles, mpvalues)
        end
    end

    # block-wise computation (BlockSpace)
    body = :(for blk in blocks
                 for p in blockspace[blk]
                     f(p, grid, particles, mpvalues)
                 end
             end)
    if schedule.value != :nothing # wrap by @threaded
        body = :(@threaded $schedule $body)
    end
    @eval function P2G(f, ::Val{$schedule}, grid::Grid, particles::AbstractVector, mpvalues::AbstractVector{<: MPValue}, blockspace::BlockSpace)
        for blocks in Sequoia.threadsafe_blocks(blockspace)
            $body
        end
    end
end

# multigrid computation (MultigridSpace)
function P2G(f, ::Val{:nothing}, grid::Grid, particles::AbstractVector, mpvalues::AbstractVector{<: MPValue}, ::MultigridSpace)
    P2G(f, Val(:nothing), grid, particles, mpvalues, nothing)
end
function P2G(f, ::Val{:static}, grid::Grid, particles::AbstractVector, mpvalues::AbstractVector{<: MPValue}, multigridspace::MultigridSpace)
    @threaded :static for p in Sequoia.eachparticleindex(particles, mpvalues)
        id = Threads.threadid()
        f(p, multigridspace.grids[id], particles, mpvalues)
    end
end
function P2G(f, ::Val{:dynamic}, grid::Grid, particles::AbstractVector, mpvalues::AbstractVector{<: MPValue}, multigridspace::MultigridSpace)
    @warn "@P2G: :dynamic schedule is not allowed for `MultigridSpace`, changed to :static" maxlog=1
    P2G(f, Val(:static), grid, particles, mpvalues, multigridspace)
end

pre_P2G(grid::Grid, particles::AbstractVector, mpvalues::AbstractVector{<: MPValue}, space, ::Val) = nothing
post_P2G(grid::Grid, particles::AbstractVector, mpvalues::AbstractVector{<: MPValue}, space, ::Val) = nothing
pre_P2G(grid::Grid, ::AbstractVector, ::AbstractVector{<: MPValue}, space::MultigridSpace, ::Val{names}) where {names} = reinit!(space, grid, Val(names))
post_P2G(grid::Grid, ::AbstractVector, ::AbstractVector{<: MPValue}, space::MultigridSpace, ::Val{names}) where {names} = add!(grid, space, Val(names))

function P2G_nosum_macro(schedule, grid_pair, nosum_equations::Vector)
    isempty(nosum_equations) && return Expr(:block)

    grid, i = unpair(grid_pair)
    foreach(ex->complete_parent_from_index!(ex, [grid=>i]), nosum_equations)

    vars = Set{Expr}()
    foreach(ex->findarrays_from_index!(vars, i, ex), nosum_equations)

    body = quote
        Sequoia.foreach_gridindex(Val($schedule), Sequoia.GridIndexStyle($(vars...)), $grid) do $i
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

"""
    @G2P grid=>i particles=>p mpvalues=>ip begin
        equations...
    end

Grid-to-particle transfer macro.

# Examples
```jl
@G2P grid=>i particles=>p mpvalues=>ip begin

    # grid-to-particle transfer
    v[p] = @∑ v[i] * N[ip]
    ∇v[p] = @∑ v[i] ⊗ ∇N[ip]

    # calculation on particle
    x[p] = x[p] + Δt * v[p]

end
```
"""
macro G2P(grid_pair, particles_pair, mpvalues_pair, equations)
    G2P_macro(QuoteNode(:nothing), grid_pair, particles_pair, mpvalues_pair, equations)
end
macro G2P(schedule::QuoteNode, grid_pair, particles_pair, mpvalues_pair, equations)
    G2P_macro(schedule, grid_pair, particles_pair, mpvalues_pair, equations)
end

function G2P_macro(schedule::QuoteNode, grid_pair, particles_pair, mpvalues_pair, equations)
    particles, p = unpair(particles_pair)
    mpvalues, _ = unpair(mpvalues_pair)

    @assert equations.head == :block
    Base.remove_linenums!(equations)
    sumornot = map(ex->issumexpr(ex, p), equations.args)
    if sort(sumornot; rev=true) != sumornot
        error("@P2G: Equations without `@∑` must come after those with `@∑`")
    end

    sum_equations = equations.args[sumornot]
    nosum_equations = equations.args[.!sumornot]

    body1 = G2P_sum_macro(grid_pair, particles_pair, mpvalues_pair, sum_equations)
    body2 = G2P_nosum_macro(particles_pair, nosum_equations)

    body = quote
        $body1
        $body2
    end

    if !DEBUG
        body = :(@inbounds $body)
    end

    if isallunderscore(mpvalues)
        iterator = :(eachindex($particles))
    else
        iterator = :(Sequoia.eachparticleindex($particles, $mpvalues))
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
    if Meta.isexpr(expr, :ref)
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
