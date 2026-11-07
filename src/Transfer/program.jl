# -----------------------------------------------------------------------------
#  Transfer programs
# -----------------------------------------------------------------------------

# ---- equations ----

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
    if !issorted(issum; rev=true)
        error("$macroname: Equations without `@∑` must come after those with `@∑`")
    end
    equations[issum], equations[.!issum]
end

# ---- weight references ----

# Weight references resolve through per-particle columns rather than a
# per-particle `BasisWeight`, whose live SubArray state spilled GPU registers.
# Each macro emits the support and column bindings before its particle stages.
# `@G2P2G` shares those bindings across both halves, including when G2P moves x[p].
struct WeightAccess
    names::Tuple
    window::Symbol
    cols::Symbol
end
WeightAccess(names; window::Symbol=gensym(:window), cols::Symbol=gensym(:wcols)) =
    WeightAccess(Tuple(names), window, cols)

function weight_columns_exprs(access::WeightAccess, weights, particles, p, grid)
    isempty(access.names) && return ()
    (:($(access.cols) = Tesserae.weight_columns($weights, Val($(access.names)), $particles, $p, Tesserae.get_mesh($grid), $(access.window))),)
end

function weight_access_exprs(access::WeightAccess, weights, particles, p, grid)
    (:($(access.window) = Tesserae.transfer_support_window($weights, $particles, $p, Tesserae.get_mesh($grid))),
     weight_columns_exprs(access, weights, particles, p, grid)...)
end

struct TrailingIndexed
    parent::Any
    trailing::Any   # particle index
    names::Any      # referenced weight properties
    cols::Any       # bound once per particle
    vals::Any       # bound once per support node
end
function TrailingIndexed(parent, trailing, access::WeightAccess)
    TrailingIndexed(parent, trailing, access.names, access.cols, gensym(:wvals))
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

# ---- macro front end ----

# The kernels are handed a grid narrowed to the referenced properties: every extra
# `SpArray` field in the argument -- referenced or not -- slows a GPU kernel down.
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

# Non-`@∑` equations run outside the support loop, where only their own stage's
# index is bound; anything else must fail at macro time, not as a runtime
# `UndefVarError` from the expansion.
function check_nosum_refs(macroname, equations, allowed, banned...)
    for index in banned
        for name in collect_transfer_refs(equations, index)
            error("$macroname: `$name[$index]` cannot be used in a non-`@∑` equation; only `[$allowed]`-indexed properties are available there")
        end
    end
end

# Particle-side sums accumulate each target into one temporary and store it once
# per equation, so a duplicated target would store what both equations
# accumulated, twice.
function check_unique_sum_targets(macroname, equations)
    seen = Set{Any}()
    for eq in equations
        eq.lhs in seen && error("$macroname: duplicate `@∑` target `$(eq.lhs)`; combine the sums into one equation")
        push!(seen, eq.lhs)
    end
end

# ---- semantic stages shared by executable and reference lowerings ----

Base.@kwdef struct TransferStages
    g2p_sum::Vector{TransferEquation} = TransferEquation[]
    p2g_sum::Vector{TransferEquation} = TransferEquation[]
    g2p_nosum::Vector{TransferEquation} = TransferEquation[]
    p2g_nosum::Vector{TransferEquation} = TransferEquation[]
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
    TransferStages(equations_g2p_sum, equations_p2g_sum, equations_g2p_nosum, equations_p2g_nosum)
end

function transfer_stages(kind::Symbol, program::TransferProgram, i, p, ip)
    stages = if kind == :G2P2G
        split_g2p2g_stages(program, i, p)
    else
        sums, nosums = split_sum_equations(program, "@$kind")
        kind == :P2G ? TransferStages(; p2g_sum=sums, p2g_nosum=nosums) :
                       TransferStages(; g2p_sum=sums, g2p_nosum=nosums)
    end
    check_nosum_refs("@$kind", stages.g2p_nosum, p, i, ip)
    check_nosum_refs("@$kind", stages.p2g_nosum, i, p, ip)
    check_unique_sum_targets("@$kind", stages.g2p_sum)
    stages
end

# Keep the mesh so the result stays a grid, and at least one array component so an
# `SpGrid` stays an `SpGrid` for dispatch and `get_spinds`.
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

# Parsing the shape the four transfer macros share keeps them from drifting, and
# lets a wrong call say what the right one looks like instead of listing `::Any`.
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

# ---- helpers ----

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
                push_unique!(scope.replacements[i],
                             :($(parent.vals) = Tesserae.weight_node_values($(parent.parent), $(parent.cols), Val($(parent.names)), $i)))
                return :($(parent.vals).$x)
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
