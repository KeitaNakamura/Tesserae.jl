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
    if !allequal(issum) && !issorted(issum; rev=true)
        error("$macroname: Equations without `@∑` must come after those with `@∑`")
    end
    equations[issum], equations[.!issum]
end

# ---- weight references ----

# Weight references resolve through per-particle columns rather than a
# per-particle `BasisWeight`, whose live SubArray state spilled GPU registers.
# `@G2P2G` shares one binding across both halves: the P2G half runs after the G2P
# half may have written `x[p]`, so rebinding would evaluate the basis at the new
# position against the window taken at the old one.
struct WeightColumnsBinding
    names::Any
    cols::Symbol
    load::Bool
end
WeightColumnsBinding(names) = WeightColumnsBinding(Tuple(names), gensym(:wcols), true)
WeightColumnsBinding(binding::WeightColumnsBinding; load::Bool) = WeightColumnsBinding(binding.names, binding.cols, load)

struct TrailingIndexed
    parent::Any
    trailing::Any   # particle index
    particles::Any
    grid::Any
    window::Any     # support index -> node
    names::Any      # referenced weight properties
    cols::Any       # bound once per particle
    vals::Any       # bound once per support node
    loadcols::Bool
end
TrailingIndexed(parent, trailing) = TrailingIndexed(parent, trailing, nothing, nothing, nothing, nothing, nothing, nothing, false)
function TrailingIndexed(parent, trailing, particles, grid, window, binding::WeightColumnsBinding)
    TrailingIndexed(parent, trailing, particles, grid, window, binding.names, binding.cols, gensym(:wvals), binding.load)
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

# `@G2P2G` fuses both loops into one pass, so the window is bound once and shared:
# whichever half holds `load=true` emits the binding statement.
struct SupportWindowBinding
    window::Symbol
    load::Bool
end
SupportWindowBinding() = SupportWindowBinding(gensym(:window), true)
SupportWindowBinding(binding::SupportWindowBinding; load::Bool) = SupportWindowBinding(binding.window, load)

function support_window_exprs(binding::SupportWindowBinding, weights, particles, p, grid)
    binding.load || return ()
    (:($(binding.window) = Tesserae.transfer_support_window($weights, $particles, $p, Tesserae.get_mesh($grid))),)
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
                # `push_unique!` emits each binding once, so the referenced
                # properties share a single basis evaluation.
                parent.loadcols && push_unique!(scope.replacements[parent.trailing],
                             :($(parent.cols) = Tesserae.weight_columns($(parent.parent), Val($(parent.names)), $(parent.particles), $(parent.trailing), Tesserae.get_mesh($(parent.grid)), $(parent.window))))
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
