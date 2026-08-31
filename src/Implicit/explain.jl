# -----------------------------------------------------------------------------
#  @explain lowering for @P2G_Matrix
# -----------------------------------------------------------------------------

function explain_P2G_Matrix(((grid_i,grid_j),(i,j)), (particles,p), ((weights_i,weights_j),(ip,jp)), partition, program::TransferProgram, threading)
    equations = program.equations
    check_matrix_program("@explain @P2G_Matrix", equations)

    bw_i, bw_j = :bw_i, :bw_j
    nodes_i, nodes_j = :nodes_i, :nodes_j
    scope = TransferScope([grid_i=>i, grid_j=>j, particles=>p, bw_i=>ip, bw_j=>jp])
    infos = matrix_infos(equations, scope, i, j)

    body = matrix_particle_body(infos, (grid_i,grid_j), (weights_i,weights_j),
                                (i,j), (ip,jp), p, (bw_i,bw_j), (nodes_i,nodes_j))
    expr_block(matrix_setup_stmts(infos, grid_i, grid_j), partitioned_particle_loop((particles,p), partition, body, threading))
end

const MATRIX_INFO_NAMES = (:table_i, :table_j, :local_matrix, :I, :J, :dofs_i, :dofs_j)
const MATRIX_INFO_SUFFIXES = (:_table_i, :_table_j, :_local, :_i, :_j, :_dofs_i, :_dofs_j)
matrix_symbols(gmat) = NamedTuple{MATRIX_INFO_NAMES}(Symbol.(gmat, MATRIX_INFO_SUFFIXES))

function matrix_infos(equations, scope, i, j)
    seen = Set{Any}()
    map(equations) do eq
        gmat, gi, gj = check_matrix_equation("@explain @P2G_Matrix", eq.lhs, i, j, seen)
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

function matrix_local_init(info, nodes_i, nodes_j)
    :($(info.local_matrix) = zeros(eltype($(info.gmat)),
        (size($(info.table_i), 1) * length($nodes_i), size($(info.table_j), 1) * length($nodes_j))))
end

matrix_idofs(info, ip) = :($(info.I) = Tesserae.local_dofs(size($(info.table_i), 1), $ip))
matrix_jdofs(info, jp) = :($(info.J) = Tesserae.local_dofs(size($(info.table_j), 1), $jp))
matrix_assign(info) = :($(info.local_matrix)[$(info.I), $(info.J)] .= $(info.rhs))
matrix_dofs(info, nodes_i, nodes_j) =
    :(($(info.dofs_i), $(info.dofs_j)) = Tesserae.support_dofs($(info.table_i), $nodes_i, $(info.table_j), $nodes_j))
matrix_add(info) = info.lhs_is_row_col ?
    :($(info.gmat)[$(info.dofs_i), $(info.dofs_j)] .+= $(info.local_matrix)) :
    :($(info.gmat)[$(info.dofs_j), $(info.dofs_i)] .+= $(info.local_matrix)')

function matrix_particle_body(infos, (grid_i,grid_j), (weights_i,weights_j), (i,j), (ip,jp), p, (bw_i,bw_j), (nodes_i,nodes_j))
    setup = matrix_particle_setup((grid_i,grid_j), (weights_i,weights_j), p, (bw_i,bw_j), (nodes_i,nodes_j))
    local_inits = map(info -> matrix_local_init(info, nodes_i, nodes_j), infos)
    local_ip = Symbol(:local_, ip)
    local_jp = Symbol(:local_, jp)
    local_idofs = map(info -> matrix_idofs(info, local_ip), infos)
    local_jdofs = map(info -> matrix_jdofs(info, local_jp), infos)
    local_assigns = map(matrix_assign, infos)
    dof_extracts = map(info -> matrix_dofs(info, nodes_i, nodes_j), infos)
    add_stmts = map(matrix_add, infos)

    inner_loop = loop_expr(:($local_ip, $ip), :(enumerate(eachindex($nodes_i))), :($i = $nodes_i[$ip]), local_idofs..., local_assigns...)
    outer_loop = loop_expr(:($local_jp, $jp), :(enumerate(eachindex($nodes_j))), :($j = $nodes_j[$jp]), local_jdofs..., inner_loop)
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
