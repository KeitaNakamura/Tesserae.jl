# -----------------------------------------------------------------------------
#  @G2P
# -----------------------------------------------------------------------------

"""
    @G2P grid=>i particles=>p weights=>ip begin
        equations...
    end

Grid-to-particle transfer macro.
Based on the `parent => index` expressions, `a[index]` in `equations`
translates to `parent.a[index]`. This `index` can be replaced with
any other name.

# Examples
```julia
@G2P grid=>i particles=>p weights=>ip begin

    # Grid-to-particle transfer
    v[p] += @∑ w[ip] * (vⁿ[i] - v[i])
    ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
    x[p] += @∑ w[ip] * v[i] * Δt

    # Calculation on particle
    Δϵₚ = symmetric(∇v[p]) * Δt
    F[p]  = (I + ∇v[p]*Δt) * F[p]
    V[p]  = V⁰[p] * det(F[p])
    σ[p] += λ*tr(Δϵₚ)*I + 2μ*Δϵₚ # Linear elastic material

end
```

This expands to roughly the following code:

```julia
# Grid-to-particle transfer
for p in eachindex(particles)
    bw = weights[p]
    nodeindices = supportnodes(bw)
    Δvₚ = zero(eltype(particles.v))
    ∇vₚ = zero(eltype(particles.∇v))
    Δxₚ = zero(eltype(particles.x))
    for ip in eachindex(nodeindices)
        i = nodeindices[ip]
        Δvₚ += bw.w[ip] * (grid.vⁿ[i] - grid.v[i])
        ∇vₚ += grid.v[i] ⊗ bw.∇w[ip]
        Δxₚ += bw.w[ip] * grid.v[i] * Δt
    end
    particles.v[p] += Δvₚ
    particles.∇v[p] = ∇vₚ
    particles.x[p] += Δxₚ
end

# Calculation on particle
for p in eachindex(particles)
    Δϵₚ = symmetric(particles.∇v[p]) * Δt
    particles.F[p]  = (I + particles.∇v[p]*Δt) * particles.F[p]
    particles.V[p]  = particles.V⁰[p] * det(particles.F[p])
    particles.σ[p] += λ*tr(Δϵₚ)*I + 2μ*Δϵₚ # Linear elastic material
end
```

Use `\$(expr)` inside transfer equations to evaluate an outer expression once
before the generated transfer loops and use the captured value in the loop body.
For example, `\$Δt` captures the current value of `Δt`.

!!! warning
    In `@G2P`, `Calculation on particles` part must be placed after
    `Grid-to-particle transfer` part.
"""
macro G2P(args...)
    schedule, grid_i, particles_p, weights_ip, _, equations = parse_transfer_macro_args("@G2P", args, false)
    G2P_expr(schedule, grid_i, particles_p, weights_ip, equations)
end

function G2P_expr(schedule::QuoteNode, grid_i::Expr, particles_p::Expr, weights_ip::Expr, equations::Expr)
    G2P_expr(schedule, unpair(grid_i), unpair(particles_p), unpair(weights_ip), parse_transfer_program(equations))
end

function G2P_expr(schedule::QuoteNode, (grid,i), (particles,p), (weights,ip), program::TransferProgram)
    sum_equations, nosum_equations = split_sum_equations(program, "@G2P")

    code = quote
        Tesserae.check_transfer_arguments("@G2P", $grid, $particles, $weights, nothing)
    end

    if !isempty(program.equations)
        body = G2P_sum_expr((grid,i), (particles,p), (weights,ip), sum_equations, nosum_equations)
        if !DEBUG
            body = :(@inbounds $body)
        end
        code = quote
            $code
            Tesserae.select_weights($weights) do w
                Tesserae.G2P(($grid, $particles, $weights, $p) -> $body, Tesserae.get_device($grid), Val($schedule), $(narrowed_grid_expr(grid, program.equations, i)), $particles, w)
            end
        end
    end

    code = interpolate_transfer_values(code, program)
    esc(prettify(code; lines=true, alias=false))
end

function G2P(f, ::CPUDevice, ::Val{scheduler}, grid, particles, weights) where {scheduler}
    tforeach(eachindex(particles), scheduler) do p
        @inline f(grid, particles, weights, p)
    end
end

function G2P_sum_expr((grid,i), (particles,p), (weights,ip), sum_equations::Vector, nosum_equations::Vector, binding::SupportWindowBinding=SupportWindowBinding(),
                      cols::Union{WeightColumnsBinding,Nothing}=nothing)
    (; window) = binding

    code = Expr(:block)
    cols = something(cols, WeightColumnsBinding(collect_transfer_refs(vcat(sum_equations, nosum_equations), ip)))
    scope = TransferScope([grid=>i, particles=>p, TrailingIndexed(weights, p, particles, grid, window, cols)=>ip]; cache=true)

    if !isempty(sum_equations)
        sum_equations = resolve_sum_equations(sum_equations, scope, "@G2P", p)
        particle_replacements = cached_replacements(scope, p)
        inner_replacements = cached_replacements(scope, i, ip)

        inits = []
        saves = []
        sum_exprs = Any[]
        for eq in sum_equations
            (; lhs, rhs, op) = eq
            tmp = Symbol(lhs, :_p)
            push!(inits, :($tmp = zero(eltype($(remove_indexing(lhs))))))
            push!(saves, Expr(op, lhs, tmp))
            push!(sum_exprs, :($tmp += $rhs))
        end

        code = quote
            $(support_window_exprs(binding, weights, particles, p, grid)...)
            $(particle_replacements...)
            $(inits...)
            for $ip in eachindex($window)
                $i = Tesserae.transfer_nodeindex($grid, $window, $ip)
                $(inner_replacements...)
                $(sum_exprs...)
            end
            $(saves...)
        end
    end

    if !isempty(nosum_equations)
        nosum_scope = TransferScope([grid=>i, particles=>p, TrailingIndexed(weights, p)=>ip])
        nosum_equations = map(eq -> Expr(eq.op, resolve_refs(eq.lhs, nosum_scope), resolve_refs(eq.rhs, nosum_scope)), nosum_equations)
        code = quote
            $code
            $(nosum_equations...)
        end
    end

    code
end
