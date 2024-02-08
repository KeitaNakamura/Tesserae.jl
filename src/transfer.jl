"""
    @P2G grid=>i particles=>p mpvalues=>ip [spspace] begin
        equations...
    end

Particle-to-grid transfer macro.

# Examples
```jl
@P2G grid=>i particles=>p mpvalues=>ip spspace begin

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
    P2G_macro(grid_pair, particles_pair, mpvalues_pair, nothing, equations)
end

macro P2G(grid_pair, particles_pair, mpvalues_pair, spspace, equations)
    P2G_macro(grid_pair, particles_pair, mpvalues_pair, spspace, equations)
end

function P2G_macro(grid_pair, particles_pair, mpvalues_pair, spspace, equations)
    _, i = unpair(grid_pair)

    @assert equations.head == :block
    Base.remove_linenums!(equations)
    sumornot = map(ex->issumexpr(ex, i), equations.args)

    sum_equations = equations.args[sumornot]
    nosum_equations = equations.args[.!sumornot]

    body1 = P2G_sum_macro(grid_pair, particles_pair, mpvalues_pair, spspace, sum_equations)
    body2 = P2G_nosum_macro(grid_pair, nosum_equations)

    quote
        $body1
        $body2
    end |> esc
end

function P2G_sum_macro(grid_pair, particles_pair, mpvalues_pair, spspace, sum_equations::Vector)
    isempty(sum_equations) && return Expr(:block)

    grid, i = unpair(grid_pair)
    particles, p = unpair(particles_pair)
    mpvalues, ip = unpair(mpvalues_pair)
    @gensym mp gridindices

    pairs = [grid=>i, particles=>p, mp=>ip]
    vars = [Set{Expr}(), Set{Expr}(), Set{Expr}()]
    foreach(ex->complete_sumeq_expr!(ex, pairs, vars), sum_equations)

    body = quote
        $(vars[2]...)
        $mp = $mpvalues[$p]
        $gridindices = neighbornodes($mp, $grid)
        @inbounds for $ip in eachindex($gridindices)
            $i = $gridindices[$ip]
            $(union(vars[1], vars[3])...)
            $(sum_equations...)
        end
    end

    if isnothing(spspace)
        body = quote
            for $p in eachindex($particles, $mpvalues)
                $body
            end
        end
    else
        body = quote
            for blocks in Sequoia.threadsafe_blocks($spspace)
                Sequoia.@threaded :dynamic for blk in blocks
                    for $p in $spspace[blk]
                        $body
                    end
                end
            end
        end
    end

    body
end

function P2G_nosum_macro(grid_pair, nosum_equations::Vector)
    isempty(nosum_equations) && return Expr(:block)
    grid, i = unpair(grid_pair)

    foreach(ex->complete_parent_from_index!(ex, [grid=>i]), nosum_equations)

    vars = Set{Expr}()
    foreach(ex->findarrays_from_index!(vars, i, ex), nosum_equations)

    body = quote
        Sequoia.foreach_gridindex(Sequoia.GridIndexStyle($(vars...)), $grid) do $i
            Base.@_inline_meta
            Base.@_propagate_inbounds_meta
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

@inline function foreach_gridindex(f, style::IndexStyle, grid::Grid)
    @inbounds @simd for i in eachindex(style, grid)
        f(i)
    end
end
@inline function foreach_gridindex(f, style::IndexCartesian, grid::SpGrid)
    @inbounds @simd for i in eachindex(style, grid)
        if isactive(grid, i)
            f(i)
        end
    end
end
@inline function foreach_gridindex(f, style::IndexSpArray, grid::SpGrid)
    @inbounds @simd for i in 1:countnnz(get_spinds(grid))
        f(UnsafeSpIndex(i))
    end
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
@P2G grid=>i particles=>p mpvalues=>ip spspace begin

    # grid-to-particle transfer
    v[p] = @∑ v[i] * N[ip]
    ∇v[p] = @∑ v[i] ⊗ ∇N[ip]

    # calculation on particle
    x[p] = x[p] + Δt * v[p]

end
```
"""
macro G2P(grid_pair, particles_pair, mpvalues_pair, equations)
    grid, i = unpair(grid_pair)
    particles, p = unpair(particles_pair)
    mpvalues, ip = unpair(mpvalues_pair)
    @gensym mp gridindices

    @assert equations.head == :block
    Base.remove_linenums!(equations)
    sumornot = map(ex->issumexpr(ex, p), equations.args)

    sum_equations = equations.args[sumornot]
    nosum_equations = equations.args[.!sumornot]

    pairs = [grid=>i, particles=>p, mp=>ip]
    vars = [Set{Expr}(), Set{Expr}(), Set{Expr}()]
    foreach(ex->complete_sumeq_expr!(ex, pairs, vars), sum_equations)

    particles_vars_declare = []
    particles_vars_store = []
    for ex in sum_equations
        lhs = ex.args[1]
        name_p = Symbol(lhs.args[1], :_p)
        push!(particles_vars_declare, :($name_p = zero(eltype($(lhs.args[1])))))
        push!(particles_vars_store, :($lhs = $name_p))
        ex.args[1] = name_p
    end

    foreach(ex->complete_parent_from_index!(ex, [particles=>p]), nosum_equations)

    body = quote
        $(vars[2]...)
        $(particles_vars_declare...)
        $mp = $mpvalues[$p]
        $gridindices = neighbornodes($mp, $grid)
        @inbounds for $ip in eachindex($gridindices)
            $i = $gridindices[$ip]
            $(union(vars[1], vars[3])...)
            $(sum_equations...)
        end
        $(particles_vars_store...)
        $(nosum_equations...)
    end

    quote
        Sequoia.@threaded :dynamic for $p in eachindex($particles, $mpvalues)
            $body
        end
    end |> esc
end

function unpair(expr::Expr)
    @assert expr.head==:call && expr.args[1]==:(=>) && isa(expr.args[2],Symbol) && isa(expr.args[3],Symbol)
    expr.args[2], expr.args[3]
end

function issumexpr(expr::Expr, index::Symbol)
    @assert expr.head==:(=) && isrefexpr(expr.args[1], index)
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

isrefexpr(expr::Expr, index::Symbol) = expr.head==:ref && expr.args[2]==index
isrefexpr(x, index) = false

function complete_sumeq_expr!(expr::Expr, pairs::Vector{Pair{Symbol, Symbol}}, vars::Vector)
    # must check `iseqexpr` in advance
    expr.args[2] = remove_∑(expr.args[2])
    expr.head = :(+=) # change `=` to `+=`
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
