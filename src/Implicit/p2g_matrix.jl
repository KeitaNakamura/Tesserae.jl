# -----------------------------------------------------------------------------
#  P2G_Matrix
# -----------------------------------------------------------------------------

# ---- support nodes ----

# Logical grid indices, not `SpGrid` storage tokens: the global DOF tables are
# built on logical indices.
@inline function matrix_support_window(weights, particles, p, grid)
    @_propagate_inbounds_meta
    window = transfer_support_window(weights, particles, p, get_mesh(grid))
    @boundscheck checkbounds(get_mesh(grid), window)
    window
end

# Kept for the readable `@explain` lowering, which materializes `BasisWeight`
# rows on purpose.
function matrix_supportnodes(bw, grid)
    @_propagate_inbounds_meta
    nodes = supportnodes(bw)
    @boundscheck checkbounds(get_mesh(grid), nodes)
    nodes, nodes
end

function matrix_supportnodes(bw_i, grid_i, bw_j, grid_j)
    @_propagate_inbounds_meta
    nodes_i = supportnodes(bw_i)
    nodes_j = supportnodes(bw_j)
    @boundscheck checkbounds(get_mesh(grid_i), nodes_i)
    @boundscheck checkbounds(get_mesh(grid_j), nodes_j)
    nodes_i, nodes_j
end

function matrix_block_supportnodes(weights, particles, particle_indices, grid)
    @_propagate_inbounds_meta
    mesh = get_mesh(grid)
    p, remaining_particles = Iterators.peel(particle_indices)
    nodes = transfer_support_window(weights, particles, p, mesh)
    first_node = first(nodes)
    last_node = last(nodes)
    for p in remaining_particles
        nodes = transfer_support_window(weights, particles, p, mesh)
        first_node = CartesianIndex(map(min, Tuple(first_node), Tuple(first(nodes))))
        last_node = CartesianIndex(map(max, Tuple(last_node), Tuple(last(nodes))))
    end
    nodes = first_node:last_node
    @boundscheck checkbounds(mesh, nodes)
    nodes
end

# ---- assembly scheduling ----

# The scatter mode rides on the assembly mode: only the particle-parallel path
# can have two writers on one stored entry, and only on GPU.
struct ParticleAssembly{S} end
ParticleAssembly() = ParticleAssembly{SerialScatter}()
struct CellAssembly end

scatter_mode(::ParticleAssembly{S}) where {S} = S()
scatter_mode(::BlockAssembly) = SerialScatter()

# -- MPM --

function P2G_Matrix(f, device::CPUDevice, schedule::Val, grids, particles, weights, partition)
    P2G((grids, particles, weights, p) -> (@inline f(grids, particles, weights, (p,), ParticleAssembly())), device, schedule, grids, particles, weights, partition)
end

function P2G_Matrix(f, ::CPUDevice, ::Val{scheduler}, grids, particles, weights, partition::Partition{<:CPUBlockStrategy}) where {scheduler}
    matrix_buffer_pool = strategy(partition).matrix_buffer_pool
    partitioned_foreach(strategy(partition), Val(scheduler)) do block
        block_particle_indices = particle_indices(partition, particles, block)
        nodes_i = matrix_block_supportnodes(weights[1], particles, block_particle_indices, grids[1])
        nodes_j = grids[1] === grids[2] && weights[1] === weights[2] ? nodes_i : matrix_block_supportnodes(weights[2], particles, block_particle_indices, grids[2])
        @inline f(grids, particles, weights, block_particle_indices, BlockAssembly(nodes_i, nodes_j, matrix_buffer_pool))
    end
end

# -- GPU --

# The shared particle-parallel kernel serves this transfer too; only the scatter
# mode differs from the CPU wrapping above.
function P2G_Matrix(f::F, device::GPUDevice, schedule::Val, grids, particles, weights, ::Nothing) where {F}
    particles = gpu_launch_collection(particles, schedule)
    backend = get_backend(device)
    kernel = gpukernel_transfer(backend)
    kernel((grids, particles, weights, p) -> (@inline f(grids, particles, weights, (p,), ParticleAssembly{AtomicScatter}())),
           grids, particles, weights; ndrange=length(particles))
end

# -- FEM and IGA --

function P2G_Matrix(f, ::CPUDevice, ::Val{scheduler}, grids, particles::QuadraturePoints,
                    weights::Tuple{<:BasisWeightArray{<:Any, <:Any, <:CellSupportMatrix}, <:BasisWeightArray{<:Any, <:Any, <:CellSupportMatrix}},
                    ::Nothing) where {scheduler}
    scheduler == :nothing || @warn "@P2G_Matrix: `Partition` must be given for threaded computation" maxlog=1

    for cell in axes(particles, 2)
        @inline f(grids, particles, weights, cell_quadrature_indices(particles, cell), CellAssembly())
    end
end

function P2G_Matrix(f, ::CPUDevice, ::Val{scheduler}, grids, particles::QuadraturePoints,
                    weights::Tuple{<:BasisWeightArray{<:Any, <:Any, <:CellSupportMatrix}, <:BasisWeightArray{<:Any, <:Any, <:CellSupportMatrix}},
                    partition::Partition{<:CellStrategy}) where {scheduler}
    partitioned_foreach(strategy(partition), Val(scheduler)) do cell
        @inline f(grids, particles, weights, particle_indices(partition, particles, cell), CellAssembly())
    end
end

function check_arguments_for_P2G_Matrix(grid, particles, weights, partition)
    check_transfer_arguments("@P2G_Matrix", grid, particles, weights, partition)
    # This macro reads the stored values directly, so deferred weights would
    # assemble from storage left unfilled -- a zero matrix, silently.
    _reject_deferred_matrix(weights)
end
_reject_deferred_matrix(weights) =
    isdeferred(weights) && error("@P2G_Matrix: cannot assemble from deferred basis weights; it reads the stored values, which deferring leaves unfilled. Call `update!` without `deferred=true` first.")
_reject_deferred_matrix(weights::Tuple) = foreach(_reject_deferred_matrix, weights)

# ---- macro implementation ----

# Shared with `@explain`, so the two cannot drift on what a valid `@P2G_Matrix`
# block is. `macroname` is threaded through only to prefix the messages.
function check_matrix_program(macroname, equations)
    isempty(equations) && error("$macroname: at least one equation is required")
    all(is_sum, equations) || error("$macroname: all equations must use `@∑`")
    nothing
end

function check_matrix_equation(macroname, lhs, i, j, seen)
    @capture(lhs, gmat_[gi_,gj_]) || error("$macroname: Invalid global matrix expression, got `$lhs`")
    ((gi == i && gj == j) || (gi == j && gj == i)) || error("$macroname: Expected expression of the form `$gmat[$i, $j]` or `$gmat[$j, $i]`, got `$lhs`")
    gmat in seen && error("$macroname: each global matrix may appear only once in a block; combine terms for `$gmat` into one `@∑` expression")
    gmat, gi, gj
end

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
macro P2G_Matrix(args...)
    P2G_Matrix_expr(parse_transfer_macro_args("@P2G_Matrix", args, true)...)
end

# -- expansion --

function P2G_Matrix_expr(schedule, grid_ij, particles_p, weights_ipjp, partition, equations)
    P2G_Matrix_expr(schedule, unpair2(grid_ij), unpair(particles_p), unpair2(weights_ipjp), partition, parse_transfer_program(equations))
end

function P2G_Matrix_expr(schedule::QuoteNode, ((grid_i,grid_j),(i,j)), (particles,p), ((weights_i,weights_j),(ip,jp)), partition, program::TransferProgram)
    @gensym grid_i′ grid_j′ weights_i′ weights_j′ gridindices_i gridindices_j particle_indices matrix_assembly remaining_particles

    equations = program.equations
    check_matrix_program("@P2G_Matrix", equations)

    # Weight references resolve through per-particle columns, like the transfer
    # macros; see the weight-references note in program.jl. With one weight set
    # on both sides the columns are bound once and shared by the row and column
    # node lookups.
    shared_weights = grid_i == grid_j && weights_i == weights_j
    names_i = collect_transfer_refs(equations, ip)
    names_j = collect_transfer_refs(equations, jp)
    cols_i = WeightColumnsBinding(shared_weights ? union(names_i, names_j) : names_i)
    cols_j = shared_weights ? WeightColumnsBinding(cols_i; load=false) : WeightColumnsBinding(names_j)
    scope = TransferScope([grid_i′=>i, grid_j′=>j, particles=>p,
                           TrailingIndexed(weights_i′, p, particles, grid_i′, gridindices_i, cols_i)=>ip,
                           TrailingIndexed(weights_j′, p, particles, grid_j′, gridindices_j, cols_j)=>jp]; cache=true)
    equations = map(equations) do eq
        TransferEquation(eq.kind, eq.lhs, resolve_refs(eq.rhs, scope), eq.op)
    end
    particle_replacements = cached_replacements(scope, p)
    i_replacements = cached_replacements(scope, i, ip)
    j_replacements = cached_replacements(scope, j, jp)
    inner_symbols = p2g_cached_symbols(cached_replacements(scope, i, j, ip, jp))

    # Shared across equations: `gmats` is the duplicate-target guard, read while
    # the records are still being built, and `hoist_exprs` collects a flat
    # cross-equation list emitted once, before the loops.
    gmats = Any[]
    hoist_exprs = Expr[]
    targets = map(equations) do equation
        (; lhs, rhs, op) = equation
        gmat, gi, gj = check_matrix_equation("@P2G_Matrix", lhs, i, j, gmats)
        push!(gmats, gmat)

        @gensym matrix buffer assembler matrix_cache

        op == :(-=) && (rhs = :(-$rhs))
        rhs = hoist_p2g_rhs!(hoist_exprs, inner_symbols, rhs)
        reorder_pair = (gi == i && gj == j) ? identity : reverse
        orientation = reorder_pair === identity ? :(Base.identity) : :(Base.reverse)
        row_grid, col_grid = reorder_pair((grid_i, grid_j))
        row_weights, col_weights = reorder_pair((weights_i, weights_j))
        dof_table_i, dof_table_j = reorder_pair((:($(assembler).row_dof_table), :($(assembler).col_dof_table)))
        assemble(f) = :(Tesserae.$f($assembler, $buffer, $matrix_assembly, $orientation, $i, $j, $ip, $jp, $rhs))
        (; matrix,
           zeroed = op == :(=),
           init = quote
               $matrix = $gmat
               $assembler = Tesserae.matrix_assembler($matrix, Tesserae.get_mesh($row_grid), Tesserae.get_mesh($col_grid), Tesserae.basis($row_weights), Tesserae.basis($col_weights))
               $matrix_cache = Tesserae.local_matrix_cache($matrix, $dof_table_i, $weights_i, $dof_table_j, $weights_j)
           end,
           # The `else` is unreachable but must stay and must throw: without it
           # `buffer` becomes a possibly-undefined local, which widens its type
           # and puts an undef check in the assembly loop.
           buffer_init = quote
               if $matrix_assembly isa Tesserae.ParticleAssembly
                   $buffer = nothing
               elseif $matrix_assembly isa Tesserae.CellAssembly
                   $buffer = Tesserae.local_matrix_buffer($matrix_cache, $dof_table_i, $gridindices_i, $dof_table_j, $gridindices_j)
               else
                   error("unknown assembly mode: $($matrix_assembly)")
               end
           end,
           block_buffer_init = :($buffer = Tesserae.block_matrix_buffer($assembler, $matrix_assembly, $orientation)),
           assemble_first = assemble(:assemble_first!),
           assemble_add = assemble(:assemble_add!),
           finish = :(Tesserae.finish_assembly!($assembler, $buffer, $matrix_assembly, $orientation)))
    end

    # Must stay a tuple literal: that is what lets `fillzero_matrix_targets!`
    # recognize a complete set of blocks and zero the parent CSC in one pass.
    zeroed_targets = [t.matrix for t in targets if t.zeroed]
    fillzero_matrix_targets = if isempty(zeroed_targets)
        nothing
    else
        :(Tesserae.fillzero_matrix_targets!(($(zeroed_targets...),)))
    end

    supportnodes_expr = if shared_weights
        quote
            $gridindices_i = Tesserae.matrix_support_window($weights_i′, $particles, $p, $grid_i′)
            $gridindices_j = $gridindices_i
        end
    else
        quote
            $gridindices_i = Tesserae.matrix_support_window($weights_i′, $particles, $p, $grid_i′)
            $gridindices_j = Tesserae.matrix_support_window($weights_j′, $particles, $p, $grid_j′)
        end
    end

    particle_init = quote
        $(particle_replacements...)
        $(hoist_exprs...)
    end

    function assemble_particle(assembly)
        quote
            for $jp in eachindex($gridindices_j)
                $j = $gridindices_j[$jp]
                $(j_replacements...)
                for $ip in eachindex($gridindices_i)
                    $i = $gridindices_i[$ip]
                    $(i_replacements...)
                    $(assembly...)
                end
            end
        end
    end

    # The cell path peels the first particle so its `assemble_first!` overwrites
    # the reused local matrix; the block path needs no peel because `acquire!`
    # returns a zeroed buffer.
    particle_or_cell_body = quote
        $p, $remaining_particles = Base.Iterators.peel($particle_indices)
        $supportnodes_expr
        $particle_init
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
            $supportnodes_expr
            $particle_init
            $(assemble_particle(map(t -> t.assemble_add, targets)).args...)
        end
        $(map(t -> t.finish, targets)...)
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
