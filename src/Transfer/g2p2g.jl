# -----------------------------------------------------------------------------
#  @G2P2G
# -----------------------------------------------------------------------------

# `@G2P2G` walks the same loop as `@P2G` on either device; only the body differs.
# `check_partition_for_transfer` rejects a device partition from every macro but
# `@P2G`, so the block-scheduled path is not reachable from here. `f` is only
# handed on, so it takes a type parameter to be specialized on.
G2P2G(f::F, device::AbstractDevice, schedule, grid, particles, weights, partition, zeroed::Tuple=()) where {F} =
    P2G(f, device, schedule, grid, particles, weights, partition, zeroed)

# The `@G2P2G` twin of `P2G_halves`: the grid-node half rides the threaded CPU
# region as its epilogue and is discharged as a separate call everywhere else.
function G2P2G_halves(f::F, device, schedule, grid, particles, weights, partition, zeroed,
                      nodebody::N, nodegrid) where {F, N}
    G2P2G(f, device, schedule, grid, particles, weights, partition, zeroed)
    P2G_nosum(nodebody, device, schedule, nodegrid)
end

function G2P2G_halves(f::F, device::CPUDevice, schedule::Val, grid, particles, weights,
                      partition::ThreadPartition, zeroed::Tuple, nodebody::N, nodegrid) where {F, N}
    epilogue = (nworkers, w) -> foreach_worker_loop(nodebody, device, nodegrid, nworkers, w)
    p2g_region(f, device, schedule, grid, particles, weights, partition, zeroed, epilogue)
end

"""
    @G2P2G grid=>i particles=>p weights=>ip [partition] begin
        equations...
    end

Combined grid-to-particle and particle-to-grid transfer macro.

Allows both [`@G2P`](@ref) (interpolation from grid to particles) and [`@P2G`](@ref) (scattering from particles to grid)
to be performed in a single loop over particles, avoiding repeated traversals.

# Examples
```julia
@G2P2G grid=>i particles=>p weights=>ip begin
    # G2P
    ∇v[p] = @∑ v[i] ⊗ ∇w[ip]

    # Particle update
    F[p] = (I + ∇v[p]*Δt) * F[p]
    σ[p] = cauchy_stress(F[p])

    # P2G
    f[i] = @∑ -V[p] * σ[p] * ∇w[ip]
end
```
"""
macro G2P2G(args...)
    G2P2G_expr(parse_transfer_macro_args("@G2P2G", args, true)...)
end

function G2P2G_expr(schedule::QuoteNode, grid_i::Expr, particles_p::Expr, weights_ip::Expr, partition, equations::Expr)
    G2P2G_expr(schedule, unpair(grid_i), unpair(particles_p), unpair(weights_ip), partition, parse_transfer_program(equations))
end

struct G2P2GStages
    g2p_sum::Vector{TransferEquation}
    p2g_sum::Vector{TransferEquation}
    g2p_nosum::Vector{TransferEquation}
    p2g_nosum::Vector{TransferEquation}
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
    G2P2GStages(equations_g2p_sum, equations_p2g_sum, equations_g2p_nosum, equations_p2g_nosum)
end

function G2P2G_expr(schedule::QuoteNode, (grid,i), (particles,p), (weights,ip), partition, program::TransferProgram)
    stages = split_g2p2g_stages(program, i, p)

    code = quote
        Tesserae.check_transfer_arguments("@G2P2G", $grid, $particles, $weights, $partition)
    end
    body = Expr(:block)
    binding = SupportWindowBinding()
    # Both halves resolve against one set of per-particle weight columns. The P2G
    # half must not rebind them: it runs after the G2P half's non-`@∑` equations,
    # which is where an explicit step writes `x[p]`.
    colsbinding = WeightColumnsBinding(union(
        collect_transfer_refs(vcat(stages.g2p_sum, stages.g2p_nosum), ip),
        collect_transfer_refs(stages.p2g_sum, ip)))

    if !isempty(stages.g2p_sum) || !isempty(stages.g2p_nosum)
        expr = G2P_sum_expr((grid,i), (particles,p), (weights,ip), stages.g2p_sum, stages.g2p_nosum, binding, colsbinding)
        body = quote
            $body
            $expr
        end
    end

    zeroed = Expr(:tuple)
    if !isempty(stages.p2g_sum)
        # `G2P_sum_expr` binds the basis weight only when it has `@∑` equations,
        # so this half must load it itself otherwise.
        p2g_binding = isempty(stages.g2p_sum) ? binding : SupportWindowBinding(binding; load=false)
        p2g_cols = isempty(stages.g2p_sum) ? colsbinding : WeightColumnsBinding(colsbinding; load=false)
        zeroed, expr = P2G_sum_expr((grid,i), (particles,p), (weights,ip), stages.p2g_sum, p2g_binding, p2g_cols)
        body = quote
            $body
            $expr
        end
    end

    if !DEBUG
        body = :(@inbounds $body)
    end
    particle_equations = vcat(collect(stages.g2p_sum), collect(stages.g2p_nosum), collect(stages.p2g_sum))
    args = [:(($grid, $particles, $weights, $p) -> $body), :(Tesserae.get_device($grid)), :(Val($schedule)),
            narrowed_grid_expr(grid, particle_equations, i), particles, :w, partition, zeroed]
    bind = nothing
    if !isempty(stages.p2g_nosum)
        @gensym nodebody nodegrid
        bind = nosum_binding_expr(nodebody, nodegrid, grid, i, stages.p2g_nosum)
        append!(args, (nodebody, nodegrid))
    end
    call = isempty(stages.p2g_nosum) ? :(Tesserae.G2P2G($(args...))) : :(Tesserae.G2P2G_halves($(args...)))
    code = quote
        $code
        $bind
        Tesserae.select_weights($weights) do w
            $call
        end
    end

    code = interpolate_transfer_values(code, program)
    esc(prettify(code; lines=true, alias=false))
end
