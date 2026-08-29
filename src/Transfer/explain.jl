import JuliaFormatter

"""
    ExplainedCode

Readable CPU reference code returned by [`@explain`](@ref).
Printing an `ExplainedCode` shows formatted code; the underlying expression is stored in `code`.
"""
struct ExplainedCode
    kind::Symbol
    code::Expr
end

Base.show(io::IO, code::ExplainedCode) = show(io, MIME("text/plain"), code)

function Base.show(io::IO, ::MIME"text/plain", code::ExplainedCode)
    println(io, "# Reference expansion of @", code.kind, ".")
    println(io, "# Runnable CPU code for understanding/debugging.")
    println(io, "# This is not the optimized lowering used by the macro.")
    println(io)
    print(io, explain_code_string(code.code))
end

"""
    @explain @P2G ...
    @explain @G2P ...
    @explain @G2P2G ...
    @explain @P2G_Matrix ...

Return readable reference code for a transfer macro.
The returned [`ExplainedCode`](@ref) stores runnable CPU reference code in `code`.
It is meant for understanding and debugging, not as a representation of the optimized
lowering used by the macro.

`@explain` supports [`@P2G`](@ref), [`@G2P`](@ref), [`@G2P2G`](@ref), [`@P2G_Matrix`](@ref),
and transfer calls prefixed with [`@threaded`](@ref).
"""
macro explain(ex)
    explained = explain_macrocall(ex)
    :(ExplainedCode($(QuoteNode(explained.kind)), $(QuoteNode(explained.code))))
end

is_line_node(x) = x isa LineNumberNode

function explain_macrocall(ex; threaded=false, schedule=QuoteNode(:nothing))
    Meta.isexpr(ex, :macrocall) || error("@explain expects a transfer macro call, got `$ex`")
    macro_name = ex.args[1]
    args = filter(!is_line_node, ex.args[2:end])
    if macro_name == Symbol("@threaded")
        return explain_threaded_call(args)
    end
    kind = Symbol(string(macro_name)[2:end])
    code = explain_transfer_call(kind, args; threaded, schedule)
    ExplainedCode(kind, readable_expr(code))
end

function explain_threaded_call(args)
    if length(args) == 1
        explain_macrocall(args[1]; threaded=true, schedule=QuoteNode(:dynamic))
    elseif length(args) == 2 && args[1] isa QuoteNode
        explain_macrocall(args[2]; threaded=true, schedule=args[1])
    else
        error("@explain @threaded expects a transfer macro call")
    end
end

readable_expr(code::Expr) = MacroTools.rmlines(prettify(code; lines=false, alias=false))

function explain_code_string(code::Expr)
    text = join(map(explain_statement_string, statement_exprs(code)), '\n')
    format_explain_code(text)
end

function explain_statement_string(stmt)
    threaded = threads_for_parts(stmt)
    if threaded === nothing
        sprint(io -> Base.show_unquoted(io, readable_expr(stmt), 0, 0))
    else
        schedule, loop = threaded
        loop_text = sprint(io -> Base.show_unquoted(io, readable_expr(loop), 0, 0))
        "Threads.@threads :$(schedule.value) $loop_text"
    end
end

format_explain_code(text::String) =
    rstrip(JuliaFormatter.format_text(text; indent=4, always_for_in=true, margin=10_000))

threads_for_parts(ex) = (@capture(ex, Threads.@threads schedule_ loop_)) ? (schedule, loop) : nothing

function explain_args(kind::Symbol, args, arities, threaded, schedule)
    if !isempty(args) && first(args) isa QuoteNode
        schedule, args = first(args), args[2:end]
    end
    length(args) in arities || error("@explain @$kind: invalid arguments")
    enabled = threaded || schedule.value != :nothing
    enabled && schedule.value == :nothing && (schedule = QuoteNode(:dynamic))
    args, (; enabled, schedule)
end

threads_loop(loop, schedule::QuoteNode) =
    Expr(:macrocall, Expr(:., :Threads, QuoteNode(Symbol("@threads"))), LineNumberNode(0, :none), schedule, loop)

maybe_threaded(loop, threading) = threading.enabled ? threads_loop(loop, threading.schedule) : loop

loop_expr(var, iter, body...) = Expr(:for, Expr(:(=), var, iter), expr_block(body))

sequential_particle_loop((particles,p), body) = loop_expr(p, :(eachindex($particles)), body)

function threaded_particle_loop((particles,p), body, threading)
    maybe_threaded(sequential_particle_loop((particles,p), body), threading)
end

function partitioned_particle_loop((particles,p), partition, body, threading)
    sequential = sequential_particle_loop((particles,p), body)
    if partition === nothing
        !threading.enabled && return sequential
        return expr_block(:(@warn "`Partition` must be given for a threaded particle-to-grid transfer" maxlog=1), sequential)
    end

    group = :group
    region = :region
    loop = loop_expr(region, group, loop_expr(p, :(particle_indices($partition, $particles, $region)), body))
    loop_expr(group, :(threadsafe_groups($partition)), maybe_threaded(loop, threading))
end

function explain_transfer_call(kind::Symbol, args; threaded=false, schedule=QuoteNode(:nothing))
    kind in (:P2G, :G2P, :G2P2G, :P2G_Matrix) || error("@explain does not support `@$kind`")
    args, threading = explain_args(kind, args, kind == :G2P ? (4,) : (4, 5), threaded, schedule)
    a, b, c, partition, equations = kind == :G2P || length(args) == 4 ? (args[1], args[2], args[3], nothing, args[4]) : args
    program = restored_program(equations)
    kind == :P2G_Matrix && return explain_P2G_Matrix(unpair2(a), unpair(b), unpair2(c), partition, program, threading)

    ctx = transfer_context(unpair(a), unpair(b), unpair(c), partition, threading)
    explain_transfer_stages(ctx, transfer_stages(kind, program, ctx))
end

function restored_program(equations::Expr)
    program = parse_transfer_program(equations)
    restore(x) = restore_interpolations(x, program)
    TransferProgram([TransferEquation(eq.kind, restore(eq.lhs), restore(eq.rhs), eq.op) for eq in program.equations],
                    Pair{Symbol, Any}[])
end

function restore_interpolations(expr, program::TransferProgram)
    isempty(program.interpolations) && return expr
    replacements = Dict(program.interpolations)
    MacroTools.postwalk(expr) do ex
        ex isa Symbol && haskey(replacements, ex) ? replacements[ex] : ex
    end
end

expr_block(stmts...) = Expr(:block, flatten_block_statements(stmts)...)

function flatten_block_statements(stmts)
    flattened = Any[]
    for stmt in stmts
        append!(flattened, statement_exprs(stmt))
    end
    flattened
end

statement_exprs(ex::Expr) = Meta.isexpr(ex, :block) ? filter(!is_line_node, ex.args) : Any[ex]
statement_exprs(ex::Union{Tuple, AbstractVector}) = flatten_block_statements(ex)

assign_expr(op::Symbol, lhs, rhs) = Expr(op, lhs, rhs)
scatter_expr(op::Symbol, lhs, rhs) = Expr(op == :(-=) ? :(-=) : :(+=), lhs, rhs)

function sum_temp(lhs)
    @capture(lhs, name_[idx_]) || error("invalid transfer LHS: $lhs")
    Symbol(name, :_sum)
end

sum_scope((grid,i), (particles,p), (bw,ip)) = TransferScope([grid=>i, particles=>p, bw=>ip])

transfer_context(grid_i, particles_p, weights_ip, partition, threading) = (; grid_i, particles_p, weights_ip, partition, threading)

function supportnode_loop((grid,i), (weights,ip), p, stmts; bw=:bw, nodes=:nodes, load_weight=true)
    prefix = load_weight ? Any[:($bw = $weights[$p]), :($nodes = supportnodes($bw, $grid))] : ()
    expr_block(prefix, loop_expr(ip, :(eachindex($nodes)), :($i = $nodes[$ip]), stmts...))
end

function explain_P2G_fillzeros((grid,i), sum_equations)
    scope = TransferScope([grid=>i])
    unique([fillzero_stmt(eq, scope) for eq in sum_equations if eq.op == :(=)])
end

fillzero_stmt(eq, scope) = :(fillzero!($(remove_indexing(resolve_refs(eq.lhs, scope)))))

function explain_P2G_grid_loop((grid,i), nosum_equations, threading)
    maybe_threaded(loop_expr(i, :(eachindex($grid)), assign_stmts(nosum_equations, TransferScope([grid=>i]))...), threading)
end

function explain_G2P_particle_body((grid,i), (particles,p), (weights,ip), sum_equations, nosum_equations)
    bw = :bw
    expr_block(
        explain_G2P_sum_body((grid,i), (particles,p), (weights,ip), (bw,ip), sum_equations),
        assign_stmts(nosum_equations, TransferScope([particles=>p])),
    )
end

function assign_stmts(equations, scope)
    map(equations) do eq
        eq = resolve_equation(eq, scope)
        assign_expr(eq.op, eq.lhs, eq.rhs)
    end
end

function explain_G2P_sum_body((grid,i), (particles,p), (weights,wp), (bw,ip), sum_equations)
    isempty(sum_equations) && return ()
    scope = sum_scope((grid,i), (particles,p), (bw,ip))
    equations = resolve_sum_equations(sum_equations, scope, "@G2P", p)
    inits, sums, saves = Any[], Any[], Any[]
    for (source_eq, eq) in zip(sum_equations, equations)
        tmp = sum_temp(source_eq.lhs)
        push!(inits, :($tmp = zero(eltype($(remove_indexing(eq.lhs))))))
        push!(sums, :($tmp += $(eq.rhs)))
        push!(saves, assign_expr(eq.op, eq.lhs, tmp))
    end
    expr_block(inits, supportnode_loop((grid,i), (weights,wp), p, sums; bw), saves)
end

transfer_stages(; g2p_sum=(), p2g_sum=(), g2p_nosum=(), p2g_nosum=()) = (; g2p_sum, p2g_sum, g2p_nosum, p2g_nosum)

function transfer_stages(kind::Symbol, program::TransferProgram, ctx)
    (_, i), (_, p), (_, ip) = ctx.grid_i, ctx.particles_p, ctx.weights_ip
    stages = if kind == :G2P2G
        split_g2p2g_stages(program, i, p)
    else
        sums, nosums = split_sum_equations(program, "@$kind")
        kind == :P2G ? transfer_stages(; p2g_sum=sums, p2g_nosum=nosums) :
                       transfer_stages(; g2p_sum=sums, g2p_nosum=nosums)
    end
    check_nosum_refs("@$kind", stages.g2p_nosum, p, i, ip)
    check_nosum_refs("@$kind", stages.p2g_nosum, i, p, ip)
    stages
end

transfer_particle_loop(ctx, body, stages) = !isempty(stages.p2g_sum) ?
    partitioned_particle_loop(ctx.particles_p, ctx.partition, body, ctx.threading) :
    threaded_particle_loop(ctx.particles_p, body, ctx.threading)

function explain_transfer_stages(ctx, stages)
    particle_body = transfer_particle_body(ctx, stages)
    particle_loop = isempty(particle_body) ? () : transfer_particle_loop(ctx, expr_block(particle_body), stages)
    grid_loop = isempty(stages.p2g_nosum) ? () : explain_P2G_grid_loop(ctx.grid_i, stages.p2g_nosum, ctx.threading)
    expr_block(explain_P2G_fillzeros(ctx.grid_i, stages.p2g_sum), particle_loop, grid_loop)
end

function transfer_particle_body(ctx, stages)
    g2p = isempty(stages.g2p_sum) && isempty(stages.g2p_nosum) ? () :
          explain_G2P_particle_body(ctx.grid_i, ctx.particles_p, ctx.weights_ip, stages.g2p_sum, stages.g2p_nosum)
    p2g = isempty(stages.p2g_sum) ? () :
          explain_P2G_sum_body(ctx.grid_i, ctx.particles_p, ctx.weights_ip, stages.p2g_sum; load_weight=isempty(stages.g2p_sum))
    flatten_block_statements((g2p, p2g))
end

function explain_P2G_sum_body((grid,i), (particles,p), (weights,ip), sum_equations; load_weight=true)
    bw = :bw
    scope = sum_scope((grid,i), (particles,p), (bw,ip))
    equations = resolve_sum_equations(sum_equations, scope, "@P2G", i)
    transfers = map(eq -> scatter_expr(eq.op, eq.lhs, eq.rhs), equations)
    supportnode_loop((grid,i), (weights,ip), p, transfers; bw, load_weight)
end
