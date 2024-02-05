macro P2G(grid_pair, particles_pair, mpvalues_pair, equations)
    P2G_macro(grid_pair, particles_pair, mpvalues_pair, nothing, equations)
end

macro P2G(grid_pair, particles_pair, mpvalues_pair, spspace, equations)
    P2G_macro(grid_pair, particles_pair, mpvalues_pair, spspace, equations)
end

function P2G_macro(grid_pair, particles_pair, mpvalues_pair, spspace, equations)
    grid, i = unpair(grid_pair)
    particles, p = unpair(particles_pair)
    mpvalues, ip = unpair(mpvalues_pair)
    @gensym mp gridindices

    @assert equations.head == :block
    Base.remove_linenums!(equations)
    all(ex->iseqexpr(ex, i), equations.args)

    pairs = [:grid=>i, :particles=>p, mp=>ip]
    vars = [Set{Expr}(), Set{Expr}(), Set{Expr}()]
    foreach(ex->complete_eqexpr!(ex, pairs, vars), equations.args)

    body = quote
        $(vars[2]...)
        $mp = $mpvalues[$p]
        $gridindices = neighbornodes($mp, $grid)
        @simd for $ip in eachindex($gridindices)
            $i = $gridindices[$ip]
            $(union(vars[1], vars[3])...)
            $equations
        end
    end
    if !DEBUG
        body = :(@inbounds $body)
    end

    if isnothing(spspace)
        quote
            for $p in eachindex($particles, $mpvalues)
                $body
            end
        end |> esc
    else
        quote
            for blocks in Sequoia.threadsafe_blocks($spspace)
                Sequoia.@threaded :dynamic for blk in blocks
                    for $p in $spspace[blk]
                        $body
                    end
                end
            end
        end |> esc
    end
end

macro G2P(grid_pair, particles_pair, mpvalues_pair, equations)
    grid, i = unpair(grid_pair)
    particles, p = unpair(particles_pair)
    mpvalues, ip = unpair(mpvalues_pair)
    @gensym mp gridindices

    @assert equations.head == :block
    Base.remove_linenums!(equations)
    all(ex->iseqexpr(ex, p), equations.args)

    pairs = [:grid=>i, :particles=>p, mp=>ip]
    vars = [Set{Expr}(), Set{Expr}(), Set{Expr}()]
    foreach(ex->complete_eqexpr!(ex, pairs, vars), equations.args)

    particles_vars_declare = []
    particles_vars_store = []
    for ex in equations.args
        lhs = ex.args[1]
        name_p = Symbol(lhs.args[1], :_p)
        push!(particles_vars_declare, :($name_p = zero(eltype($(lhs.args[1])))))
        push!(particles_vars_store, :($lhs = $name_p))
        ex.args[1] = name_p
    end

    body = quote
        $(vars[2]...)
        $(particles_vars_declare...)
        $mp = $mpvalues[$p]
        $gridindices = neighbornodes($mp, $grid)
        @simd for $ip in eachindex($gridindices)
            $i = $gridindices[$ip]
            $(union(vars[1], vars[3])...)
            $equations
        end
        $(particles_vars_store...)
    end
    if !DEBUG
        body = :(@inbounds $body)
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

iseqexpr(expr::Expr, index::Symbol) = expr.head==:(=) && isrefexpr(expr.args[1], index) && issumexpr(expr.args[2])
iseqexpr(x, index) = false

issumexpr(expr::Expr) = expr.head==:macrocall && length(expr.args)==3 && isa(expr.args[2],LineNumberNode)
issumexpr(x) = false

isrefexpr(expr::Expr, index::Symbol) = expr.head==:ref && expr.args[2]==index
isrefexpr(x, index) = false

function complete_eqexpr!(expr::Expr, pairs::Vector{Pair{Symbol, Symbol}}, vars::Vector)
    # must check `iseqexpr` in advance
    expr.args[2] = remove_∑(expr.args[2])
    expr.head = :(+=) # change `=` to `+=`
    complete_lhseqexpr!(expr.args[1], pairs)
    complete_rhseqexpr!(Meta.quot(expr.args[2]), pairs, vars)
end

function remove_∑(rhs::Expr)
    rhs.args[3] # extract only inside of @∑ (i.e., remove @∑)
end

function complete_lhseqexpr!(lhs::Expr, pairs::Vector{Pair{Symbol, Symbol}})
    for p in pairs
        if p.second == lhs.args[2] # same index
            lhs.args[1] = :($(p.first).$(lhs.args[1]))
        end
    end
end

function complete_rhseqexpr!(expr::Expr, pairs::Vector{Pair{Symbol, Symbol}}, vars::Vector)
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
        complete_rhseqexpr!(ex, pairs, vars)
    end
end
complete_rhseqexpr!(expr, pairs::Vector{Pair{Symbol, Symbol}}, vars::Vector) = nothing
