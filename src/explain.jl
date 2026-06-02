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

loop_expr(var, iter, body...) = Expr(:for, Expr(:(=), var, iter), expr_block(body))

sequential_particle_loop((particles,p), body) = loop_expr(p, :(eachindex($particles)), body)

function threaded_particle_loop((particles,p), body, threading)
    loop = sequential_particle_loop((particles,p), body)
    threading.enabled ? threads_loop(loop, threading.schedule) : loop
end

function partitioned_particle_loop((particles,p), partition, body, threading)
    sequential = sequential_particle_loop((particles,p), body)
    if partition === nothing
        !threading.enabled && return sequential
        return expr_block(:(@warn "@P2G: `ThreadPartition` must be given for threaded computation" maxlog=1), sequential)
    end

    group = :group
    region = :region
    loop = loop_expr(region, group, loop_expr(p, :(particle_indices($partition, $particles, $region)), body))
    inner = threading.enabled ? threads_loop(loop, threading.schedule) : loop
    loop_expr(group, :(threadsafe_groups($partition)), inner)
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

function explain_P2G_grid_loop((grid,i), nosum_equations)
    loop_expr(i, :(eachindex($grid)), assign_stmts(nosum_equations, TransferScope([grid=>i]))...)
end

function explain_G2P_particle_body((grid,i), (particles,p), (weights,ip), sum_equations, nosum_equations)
    bw = :bw
    expr_block(
        explain_G2P_sum_body((grid,i), (particles,p), (weights,ip), (bw,ip), sum_equations),
        assign_stmts(nosum_equations, sum_scope((grid,i), (particles,p), (bw,ip))),
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
    if kind == :G2P2G
        (_, i), (_, p) = ctx.grid_i, ctx.particles_p
        return split_g2p2g_stages(program, i, p)
    end
    sums, nosums = split_sum_equations(program, "@$kind")
    kind == :P2G ? transfer_stages(; p2g_sum=sums, p2g_nosum=nosums) :
                   transfer_stages(; g2p_sum=sums, g2p_nosum=nosums)
end

transfer_particle_loop(ctx, body, stages) = !isempty(stages.p2g_sum) ?
    partitioned_particle_loop(ctx.particles_p, ctx.partition, body, ctx.threading) :
    threaded_particle_loop(ctx.particles_p, body, ctx.threading)

function explain_transfer_stages(ctx, stages)
    particle_body = transfer_particle_body(ctx, stages)
    particle_loop = isempty(particle_body) ? () : transfer_particle_loop(ctx, expr_block(particle_body), stages)
    grid_loop = isempty(stages.p2g_nosum) ? () : explain_P2G_grid_loop(ctx.grid_i, stages.p2g_nosum)
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

function explain_P2G_Matrix(((grid_i,grid_j),(i,j)), (particles,p), ((weights_i,weights_j),(ip,jp)), partition, program::TransferProgram, threading)
    equations = program.equations
    isempty(equations) && error("@explain @P2G_Matrix: at least one equation is required")
    all(is_sum, equations) || error("@explain @P2G_Matrix: all equations must use `@∑`")

    bw_i, bw_j = :bw_i, :bw_j
    nodes_i, nodes_j = :nodes_i, :nodes_j
    scope = TransferScope([grid_i=>i, grid_j=>j, particles=>p, bw_i=>ip, bw_j=>jp])
    infos = matrix_infos(equations, scope, i, j)

    body = matrix_particle_body(infos, (grid_i,grid_j), (weights_i,weights_j),
                                (i,j), (ip,jp), p, (bw_i,bw_j), (nodes_i,nodes_j))
    expr_block(matrix_setup_stmts(infos, grid_i, grid_j), partitioned_particle_loop((particles,p), partition, body, threading))
end

const MATRIX_INFO_NAMES = (:table_i, :table_j, :local_i, :local_j, :local_matrix, :I, :J, :dofs_i, :dofs_j)
const MATRIX_INFO_SUFFIXES = (:_table_i, :_table_j, :_local_i, :_local_j, :_local, :_i, :_j, :_dofs_i, :_dofs_j)
matrix_symbols(gmat) = NamedTuple{MATRIX_INFO_NAMES}(Symbol.(gmat, MATRIX_INFO_SUFFIXES))

function matrix_infos(equations, scope, i, j)
    seen = Set{Any}()
    map(equations) do eq
        @capture(eq.lhs, gmat_[gi_,gj_]) || error("@explain @P2G_Matrix: invalid matrix LHS: $(eq.lhs)")
        ((gi == i && gj == j) || (gi == j && gj == i)) || error("@explain @P2G_Matrix: invalid matrix LHS index: $(eq.lhs)")
        gmat in seen && error("@explain @P2G_Matrix: each global matrix may appear only once in a block")
        push!(seen, gmat)

        rhs = resolve_refs(eq.rhs, scope)
        eq.op == :(-=) && (rhs = :(-($rhs)))
        (
            matrix_symbols(gmat)...,
            gmat = gmat,
            rhs = rhs,
            lhs_is_row_col = gi == i && gj == j,
            fillzero = eq.op == :(=),
        )
    end
end

matrix_setup_stmts(infos, grid_i, grid_j) = map(infos) do info
    expr_block(info.fillzero ? :(fillzero!($(info.gmat))) : (), matrix_table_stmt(info, grid_i, grid_j))
end

matrix_table_stmt(info, grid_i, grid_j) = info.lhs_is_row_col ?
    :(($(info.table_i), $(info.table_j)) = Tesserae.matrix_dof_tables($(info.gmat), $grid_i, $grid_j)) :
    :(($(info.table_j), $(info.table_i)) = Tesserae.matrix_dof_tables($(info.gmat), $grid_j, $grid_i))

matrix_local_init(info, nodes_i, nodes_j) = quote
    $(info.local_i) = Tesserae.local_dof_table($(info.table_i), $nodes_i)
    $(info.local_j) = Tesserae.local_dof_table($(info.table_j), $nodes_j)
    $(info.local_matrix) = zeros(eltype($(info.gmat)), (length($(info.local_i)), length($(info.local_j))))
end

matrix_idofs(info, ip) = :($(info.I) = Tesserae.local_dofs($(info.local_i), $ip))
matrix_jdofs(info, jp) = :($(info.J) = Tesserae.local_dofs($(info.local_j), $jp))
matrix_assign(info) = :($(info.local_matrix)[$(info.I), $(info.J)] .= $(info.rhs))
matrix_dofs(info, nodes_i, nodes_j) =
    :(($(info.dofs_i), $(info.dofs_j)) = Tesserae.support_dofs($(info.table_i), $nodes_i, $(info.table_j), $nodes_j))
matrix_add(info) = info.lhs_is_row_col ?
    :($(info.gmat)[$(info.dofs_i), $(info.dofs_j)] .+= $(info.local_matrix)) :
    :($(info.gmat)[$(info.dofs_j), $(info.dofs_i)] .+= $(info.local_matrix)')

function matrix_particle_body(infos, (grid_i,grid_j), (weights_i,weights_j), (i,j), (ip,jp), p, (bw_i,bw_j), (nodes_i,nodes_j))
    setup = matrix_particle_setup((grid_i,grid_j), (weights_i,weights_j), p, (bw_i,bw_j), (nodes_i,nodes_j))
    local_inits = map(info -> matrix_local_init(info, nodes_i, nodes_j), infos)
    local_idofs = map(info -> matrix_idofs(info, ip), infos)
    local_jdofs = map(info -> matrix_jdofs(info, jp), infos)
    local_assigns = map(matrix_assign, infos)
    dof_extracts = map(info -> matrix_dofs(info, nodes_i, nodes_j), infos)
    add_stmts = map(matrix_add, infos)

    inner_loop = loop_expr(ip, :(eachindex($nodes_i)), :($i = $nodes_i[$ip]), local_idofs..., local_assigns...)
    outer_loop = loop_expr(jp, :(eachindex($nodes_j)), :($j = $nodes_j[$jp]), local_jdofs..., inner_loop)
    expr_block(setup, local_inits, outer_loop, dof_extracts, add_stmts)
end

function matrix_particle_setup((grid_i,grid_j), (weights_i,weights_j), p, (bw_i,bw_j), (nodes_i,nodes_j))
    same_weights = weights_i == weights_j
    rhs_j = same_weights ? bw_i : :($weights_j[$p])
    node_stmt = if grid_i == grid_j && same_weights
        :(($nodes_i, $nodes_j) = Tesserae.matrix_supportnodes($bw_i, $grid_i))
    else
        :(($nodes_i, $nodes_j) = Tesserae.matrix_supportnodes($bw_i, $grid_i, $bw_j, $grid_j))
    end
    expr_block(:($bw_i = $weights_i[$p]), :($bw_j = $rhs_j), node_stmt)
end
