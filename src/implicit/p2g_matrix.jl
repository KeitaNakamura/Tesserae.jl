# -----------------------------------------------------------------------------
#  P2G_Matrix
# -----------------------------------------------------------------------------

# ---- support nodes ----

function matrix_supportnodes(bw, grid)
    @_propagate_inbounds_meta
    # Matrix assembly indexes global DOF tables, which are built on logical
    # grid indices. For an SpGrid, supportnodes(bw, grid) returns SpIndex
    # storage tokens instead. Using those here would require SpIndex to fully
    # support AbstractArray indexing.
    nodes = supportnodes(bw)
    @boundscheck checkbounds(get_mesh(grid), nodes)
    nodes, nodes
end

function matrix_supportnodes(bw_i, grid_i, bw_j, grid_j)
    @_propagate_inbounds_meta
    # See the single-grid method: matrix DOF tables need logical grid indices,
    # not SpGrid storage tokens.
    nodes_i = supportnodes(bw_i)
    nodes_j = supportnodes(bw_j)
    @boundscheck checkbounds(get_mesh(grid_i), nodes_i)
    @boundscheck checkbounds(get_mesh(grid_j), nodes_j)
    nodes_i, nodes_j
end

function matrix_block_supportnodes(weights, particle_indices, grid)
    @_propagate_inbounds_meta
    p, remaining_particles = Iterators.peel(particle_indices)
    nodes = supportnodes(weights[p])
    first_node = first(nodes)
    last_node = last(nodes)
    for p in remaining_particles
        nodes = supportnodes(weights[p])
        first_node = CartesianIndex(map(min, Tuple(first_node), Tuple(first(nodes))))
        last_node = CartesianIndex(map(max, Tuple(last_node), Tuple(last(nodes))))
    end
    nodes = first_node:last_node
    @boundscheck checkbounds(get_mesh(grid), nodes)
    nodes
end

# ---- assembly scheduling ----

struct ParticleAssembly end
struct CellAssembly end

# -- MPM --

function P2G_Matrix(f, device::CPUDevice, schedule::Val, grids, particles, weights, partition)
    P2G((grids, particles, weights, p) -> (@inline f(grids, particles, weights, (p,), ParticleAssembly())), device, schedule, grids, particles, weights, partition)
end

function P2G_Matrix(f, ::CPUDevice, ::Val{scheduler}, grids, particles, weights, partition::ThreadPartition{<:BlockStrategy}) where {scheduler}
    matrix_buffer_pool = strategy(partition).matrix_buffer_pool
    for group in threadsafe_groups(partition)
        tforeach(group, scheduler) do block
            block_particle_indices = particle_indices(partition, particles, block)
            nodes_i = matrix_block_supportnodes(weights[1], block_particle_indices, grids[1])
            nodes_j = grids[1] === grids[2] && weights[1] === weights[2] ? nodes_i : matrix_block_supportnodes(weights[2], block_particle_indices, grids[2])
            @inline f(grids, particles, weights, block_particle_indices, BlockAssembly(nodes_i, nodes_j, matrix_buffer_pool))
        end
    end
end

# -- FEM and IGA --

function P2G_Matrix(f, ::CPUDevice, ::Val{scheduler}, grids, particles::QuadraturePoints,
                    weights::Tuple{<:BasisWeightArray{<:Any, <:Any, <:CellSupportMatrix}, <:BasisWeightArray{<:Any, <:Any, <:CellSupportMatrix}},
                    ::Nothing) where {scheduler}
    scheduler == :nothing || @warn "@P2G_Matrix: `ThreadPartition` must be given for threaded computation" maxlog=1

    for cell in axes(particles, 2)
        particle_indices = (CartesianIndex(q, cell) for q in axes(particles, 1))
        @inline f(grids, particles, weights, particle_indices, CellAssembly())
    end
end

function P2G_Matrix(f, ::CPUDevice, ::Val{scheduler}, grids, particles::QuadraturePoints,
                    weights::Tuple{<:BasisWeightArray{<:Any, <:Any, <:CellSupportMatrix}, <:BasisWeightArray{<:Any, <:Any, <:CellSupportMatrix}},
                    partition::ThreadPartition{<:CellStrategy}) where {scheduler}
    for group in threadsafe_groups(partition)
        tforeach(group, scheduler) do cell
            @inline f(grids, particles, weights, particle_indices(partition, particles, cell), CellAssembly())
        end
    end
end

function check_arguments_for_P2G_Matrix(grid, particles, weights, partition)
    check_arguments_for_P2G(grid, particles, weights, partition)
    @assert get_device(grid) isa CPUDevice
end

# ---- macro implementation ----

# -- matrix zeroing --

function sparse_matrix_blocks_cover_parent(blocks, matrix)
    all(block -> parent(block) === matrix, blocks) || return false
    sum(nnz, blocks) == nnz(matrix) || return false
    for j in 2:length(blocks), i in 1:j-1
        block_i = blocks[i]
        block_j = blocks[j]
        isdisjoint(block_i.rows, block_j.rows) || isdisjoint(block_i.cols, block_j.cols) || return false
    end
    true
end

fillzero_matrix_targets!(matrices) = foreach(fillzero!, matrices)

# A complete set of disjoint blocks can zero its parent CSC in one contiguous pass.
function fillzero_matrix_targets!(blocks::Tuple{Vararg{SparseMatrixBlockView}})
    isempty(blocks) && return nothing
    matrix = parent(first(blocks))
    if sparse_matrix_blocks_cover_parent(blocks, matrix)
        fillzero!(matrix)
    else
        foreach(fillzero!, blocks)
    end
    nothing
end

# -- public macro --

"""
    @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) [partition] begin
        equations...
    end

Particle-to-grid transfer macro for assembling a global matrix.
A typical global stiffness matrix can be assembled as follows:

```julia
@P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
    K[i,j] = @∑ ∇w[ip] ⊡ c[p] ⊡ ∇w[jp] * V[p]
end
```

where `c` and `V` denote the stiffness (symmetric fourth-order) tensor and the volume, respectively.
It is recommended to create global stiffness `K` using [`create_sparse_matrix`](@ref).
Individual views returned by [`create_block_sparse_matrix`](@ref) may also be
used as targets, for example `blocks[1,2][i,j]`.
"""
macro P2G_Matrix(grid_ij, particles_p, weights_ipjp, equations)
    P2G_Matrix_expr(QuoteNode(:nothing), grid_ij, particles_p, weights_ipjp, nothing, equations)
end
macro P2G_Matrix(grid_ij, particles_p, weights_ipjp, partition, equations)
    P2G_Matrix_expr(QuoteNode(:nothing), grid_ij, particles_p, weights_ipjp, partition, equations)
end
macro P2G_Matrix(schedule::QuoteNode, grid_ij, particles_p, weights_ipjp, equations)
    P2G_Matrix_expr(schedule, grid_ij, particles_p, weights_ipjp, nothing, equations)
end
macro P2G_Matrix(schedule::QuoteNode, grid_ij, particles_p, weights_ipjp, partition, equations)
    P2G_Matrix_expr(schedule, grid_ij, particles_p, weights_ipjp, partition, equations)
end

# -- expansion --

function P2G_Matrix_expr(schedule, grid_ij, particles_p, weights_ipjp, partition, equations)
    P2G_Matrix_expr(schedule, unpair2(grid_ij), unpair(particles_p), unpair2(weights_ipjp), partition, parse_transfer_program(equations))
end

function P2G_Matrix_expr(schedule::QuoteNode, ((grid_i,grid_j),(i,j)), (particles,p), ((weights_i,weights_j),(ip,jp)), partition, program::TransferProgram)
    @gensym grid_i′ grid_j′ weights_i′ weights_j′ bw_i bw_j gridindices_i gridindices_j particle_indices matrix_assembly remaining_particles

    equations = program.equations
    isempty(equations) && error("@P2G_Matrix: at least one equation is required")
    all(is_sum, equations) || error("@P2G_Matrix: all equations must use `@∑`")

    scope = TransferScope([grid_i′=>i, grid_j′=>j, particles=>p, bw_i=>ip, bw_j=>jp]; cache=true)
    equations = map(equations) do eq
        TransferEquation(eq.kind, eq.lhs, resolve_refs(eq.rhs, scope), eq.op)
    end
    particle_replacements = cached_replacements(scope, p)
    i_replacements = cached_replacements(scope, i, ip)
    j_replacements = cached_replacements(scope, j, jp)
    inner_symbols = p2g_cached_symbols(cached_replacements(scope, i, j, ip, jp))

    # One record per matrix target. `gmats` and `hoist_exprs` stay shared across
    # equations: the first is the duplicate-target guard, read while the records
    # are still being built, and the second collects a flat cross-equation list
    # that is emitted once, before the loops.
    gmats = Any[]
    hoist_exprs = Expr[]
    targets = map(equations) do equation
        (; lhs, rhs, op) = equation
        @capture(lhs, gmat_[gi_,gj_]) || error("@P2G_Matrix: Invalid global matrix expression, got `$lhs`")
        ((gi == i && gj == j) || (gi == j && gj == i)) || error("@P2G_Matrix: Expected expression of the form `$gmat[$i, $j]` or `$gmat[$j, $i]`, got `$lhs`")
        gmat in gmats && error("@P2G_Matrix: each global matrix may appear only once in a block; combine terms for `$gmat` into one `@∑` expression")
        push!(gmats, gmat)

        @gensym matrix buffer assembler matrix_cache I J

        op == :(-=) && (rhs = :(-$rhs))
        rhs = hoist_p2g_rhs!(hoist_exprs, inner_symbols, rhs)
        reorder_pair = (gi == i && gj == j) ? identity : reverse
        orientation = reorder_pair === identity ? :(Base.identity) : :(Base.reverse)
        row_grid, col_grid = reorder_pair((grid_i, grid_j))
        row_weights, col_weights = reorder_pair((weights_i, weights_j))
        dof_table_i, dof_table_j = reorder_pair((:($(assembler).row_dof_table), :($(assembler).col_dof_table)))
        assemble(f) = :(Tesserae.$f($assembler, $buffer, $orientation, $i, $j, $I, $J, $rhs))
        (; matrix,
           zeroed = op == :(=),
           init = quote
               $matrix = $gmat
               $assembler = Tesserae.matrix_assembler($matrix, Tesserae.get_mesh($row_grid), Tesserae.get_mesh($col_grid), Tesserae.basis($row_weights), Tesserae.basis($col_weights))
               $matrix_cache = Tesserae.local_matrix_cache($matrix, $dof_table_i, $weights_i, $dof_table_j, $weights_j)
           end,
           buffer_init = quote
               if $matrix_assembly isa Tesserae.ParticleAssembly
                   $buffer = nothing
               elseif $matrix_assembly isa Tesserae.CellAssembly
                   $buffer = Tesserae.local_matrix_buffer($matrix_cache, $dof_table_i, $gridindices_i, $dof_table_j, $gridindices_j)
               else
                   error("BlockAssembly must use block assembly")
               end
           end,
           block_buffer_init = :($buffer = Tesserae.block_matrix_buffer($assembler, $matrix_assembly, $orientation)),
           jdof = :($J = Tesserae.matrix_col_dofs($buffer, $jp)),
           idof = :($I = Tesserae.matrix_row_dofs($buffer, $ip)),
           assemble_first = assemble(:assemble_first!),
           assemble_add = assemble(:assemble_add!),
           assemble_block_entry = :(Tesserae.assemble_block_entry!($assembler, $buffer, $matrix_assembly, $orientation, $i, $j, $rhs)),
           finish = :(Tesserae.finish_assembly!($assembler, $buffer, $orientation, $gridindices_i, $gridindices_j)),
           finish_block = :(Tesserae.finish_block_assembly!($assembler, $buffer, $matrix_assembly, $orientation)))
    end

    local_jdofs = map(t -> t.jdof, targets)
    local_idofs = map(t -> t.idof, targets)

    # Must stay a tuple literal: that is what lets `fillzero_matrix_targets!`
    # recognize a complete set of blocks and zero the parent CSC in one pass.
    zeroed_targets = [t.matrix for t in targets if t.zeroed]
    fillzero_matrix_targets = if isempty(zeroed_targets)
        nothing
    else
        :(Tesserae.fillzero_matrix_targets!(($(zeroed_targets...),)))
    end

    supportnodes_expr = if grid_i == grid_j && weights_i == weights_j
        :(($gridindices_i, $gridindices_j) = Tesserae.matrix_supportnodes($bw_i, $grid_i′))
    else
        :(($gridindices_i, $gridindices_j) = Tesserae.matrix_supportnodes($bw_i, $grid_i′, $bw_j, $grid_j′))
    end

    particle_init = quote
        $(particle_replacements...)
        $(hoist_exprs...)
        $bw_i, $bw_j = $weights_i′[$p], $weights_j′[$p]
    end

    # The block path passes no dof lists: its `buffer` is a `BlockMatrixBuffer`,
    # for which `matrix_row_dofs`/`matrix_col_dofs` have no method on purpose --
    # block entries address the buffer by node, not by local dof range.
    function assemble_particle(assembly, jdofs=local_jdofs, idofs=local_idofs)
        quote
            for $jp in eachindex($gridindices_j)
                $j = $gridindices_j[$jp]
                $(j_replacements...)
                $(jdofs...)
                for $ip in eachindex($gridindices_i)
                    $i = $gridindices_i[$ip]
                    $(i_replacements...)
                    $(idofs...)
                    $(assembly...)
                end
            end
        end
    end

    particle_or_cell_body = quote
        $p, $remaining_particles = Base.Iterators.peel($particle_indices)
        $particle_init
        $supportnodes_expr
        $(map(t -> t.buffer_init, targets)...)
        $(assemble_particle(map(t -> t.assemble_first, targets)))
        for $p in $remaining_particles
            $particle_init
            $(assemble_particle(map(t -> t.assemble_add, targets)))
        end
        $(map(t -> t.finish, targets)...)
    end

    block_body = quote
        $(map(t -> t.block_buffer_init, targets)...)
        for $p in $particle_indices
            $particle_init
            $supportnodes_expr
            $(assemble_particle(map(t -> t.assemble_block_entry, targets), (), ()).args...)
        end
        $(map(t -> t.finish_block, targets)...)
    end

    body = quote
        if $matrix_assembly isa Tesserae.BlockAssembly
            $block_body
        else
            $particle_or_cell_body
        end
    end

    if !DEBUG
        body = :(@inbounds $body)
    end

    body = quote
        let
            $check_arguments_for_P2G_Matrix($grid_i, $particles, $weights_i, $partition)
            $check_arguments_for_P2G_Matrix($grid_j, $particles, $weights_j, $partition)
            $(map(t -> t.init, targets)...)
            $fillzero_matrix_targets
            Tesserae.P2G_Matrix((($grid_i′,$grid_j′), $particles, ($weights_i′,$weights_j′), $particle_indices, $matrix_assembly) -> $body,
                                $get_device($grid_i), Val($schedule), ($grid_i,$grid_j), $particles, ($weights_i,$weights_j), $partition)
        end
    end

    esc(interpolate_transfer_values(body, program))
end

# Like `unpair`, but the LHS is always a pair: a single parent is shared by both
# indices. Only the two-index RHS forms are valid here.
function unpair2(ex::Expr)
    lhs, rhs = unpair(ex)
    rhs isa Tuple || error("invalid expression, $ex")
    lhs isa Tuple ? (lhs, rhs) : ((lhs, lhs), rhs)
end
