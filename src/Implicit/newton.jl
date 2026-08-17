"""
    Tesserae.newton!(x::AbstractVector, f, J,
                     maxiter = 100, atol = zero(eltype(x)), rtol = sqrt(eps(eltype(x))),
                     linsolve = (x,A,b) -> copyto!(x, A\\b),
                     backtracking = false, verbose = false)

A simple implementation of Newton's method.
The functions `f(x)` and `J(x)` should return the residual vector and its Jacobian, respectively.

Evaluation order:

```julia
r = f(x)              # update state/caches derived from x and return residual
while not converged
    x_old = x
    Jx = J(x)         # compute from x or reuse caches from f(x)
    δx = solve(Jx, r)

    if backtracking
        ϕ′0 = -dot(r, Jx, δx)
        ϕ′0 < 0 || fail
        for α in trial_steps
            x = x_old - α * δx
            r = f(x)  # update trial state
            accept && break
        end
    else
        x = x_old - δx
        r = f(x)
    end
end
```

If backtracking fails, `x` is restored to the last accepted iterate and `f(x)` is called once more to restore the corresponding state.

!!! tip
    At each iteration, `newton!` evaluates `J(x)` only after `f(x)` has already been evaluated at the same `x`.
    In simulation codes, residual and tangent/Jacobian assembly often share intermediate quantities.
    These quantities may be stored in caller-owned state while evaluating `f(x)`, so that the following `J(x)` call can reuse them without recomputing them.
    This is optional: `J(x)` may also assemble the Jacobian directly from `x`.
"""
function newton!(
        x::AbstractVector, f, J;
        maxiter::Int=100, atol::Real=zero(eltype(x)), rtol::Real=sqrt(eps(eltype(x))),
        linsolve=(x,A,b)->copyto!(x,A\b), backtracking::Bool=false, verbose::Bool=false)

    T = eltype(x)

    r = f(x)
    rnorm = rnorm0 = norm(r)
    δx = similar(x)

    # old accepted step values
    x_old, rnorm_old = similar(x), rnorm

    iter = 0
    solved = rnorm0 ≤ atol
    giveup = !isfinite(rnorm)

    if verbose
        newton_print_header(maxiter, atol, rtol)
        newton_print_row(maxiter, iter, rnorm, newton_residual_ratio(rnorm, rnorm0))
    end

    while !(solved || giveup)
        @. x_old = x
        rnorm_old = rnorm

        Jx = J(x)
        linsolve(fillzero!(δx), Jx, r)

        if backtracking
            ϕ0 = rnorm_old * rnorm_old / 2
            ϕ′0 = -dot(r, Jx, δx)
            if !(isfinite(ϕ′0) && ϕ′0 < 0)
                giveup = true
                break
            end
            accepted = newton_backtracking(one(T), ϕ0, ϕ′0) do α
                @. x = x_old - α * δx # update `x`
                r .= f(x) # update r in backtracking process
                y = norm(r)
                y * y / 2
            end
            if !accepted
                @. x = x_old
                f(x) # restore state derived from x_old
                giveup = true
                break
            end
        else
            @. x = x_old - δx
            r .= f(x)
        end

        rnorm = norm(r)
        solved = rnorm ≤ max(atol, rtol*rnorm0)
        iter += 1
        giveup = !isfinite(rnorm) || iter ≥ maxiter

        verbose && newton_print_row(maxiter, iter, rnorm, newton_residual_ratio(rnorm, rnorm0))
    end
    verbose && println()

    solved
end

newton_residual_ratio(rnorm, rnorm0) = iszero(rnorm0) ? zero(rnorm0) : rnorm/rnorm0

function newton_print_header(maxiter, atol, rtol)
    n = ndigits(maxiter)
    @printf(" # ≤ %d  f ≤ %-8.2e  f/f₀ ≤ %-8.2e\n", maxiter, atol, rtol)
    @printf(" %s  %s  %s\n", "─"^(4+n), "─"^12, "─"^15)
end
function newton_print_row(maxiter, iter, f, f_f0)
    n = ndigits(maxiter)
    @printf(" %s%s  %12.2e  %15.2e\n", " "^4, lpad(iter, n), f, f_f0)
end

function newton_backtracking(ϕ, α::T, ϕ0::T, ϕ′0::T; c::T = T(1e-4), ρ_hi::T = T(0.5), ρ_lo::T = T(0.1), maxiter::Int=1000) where {T <: Real}
    @assert 0 < ρ_lo < ρ_hi < 1
    local α_prev, ϕα_prev
    for trial in 1:maxiter
        ϕα = ϕ(α)
        ϕα ≤ ϕ0 + c*α*ϕ′0 && return true
        abs(α) < eps(T)^T(2/3) && return false

        if trial == 1
            α_new = quad_step(α, ϕα, ϕ0, ϕ′0, ρ_hi, ρ_lo)
        else
            α_new = cubic_step(α, ϕα, α_prev, ϕα_prev, ϕ0, ϕ′0, ρ_hi, ρ_lo)
        end
        α_new = clamp(α_new, α*ρ_lo, α*ρ_hi)
        α_prev, ϕα_prev = α, ϕα
        α = α_new
    end
    false
end

function quad_step(α, ϕα, ϕ0, ϕ′0, ρ_hi, ρ_lo)
    den = 2(ϕα - α*ϕ′0 - ϕ0)
    if isfinite(den) && den > 0
        return -α^2 * ϕ′0 / den
    else
        return ρ_lo * α
    end
end

function cubic_step(α, ϕα, α_prev, ϕα_prev, ϕ0, ϕ′0, ρ_hi, ρ_lo)
    den = α_prev^2 * α^2 * (α - α_prev)
    if isfinite(den) && !iszero(den)
        sα = ϕα - ϕ0 - ϕ′0*α
        sα_prev = ϕα_prev - ϕ0 - ϕ′0*α_prev
        a = ( α_prev^2 * sα - α^2 * sα_prev) / den
        b = (-α_prev^3 * sα + α^3 * sα_prev) / den

        !(isfinite(a) && isfinite(b)) && return ρ_lo * α

        # quadratic
        if abs(a) ≤ eps(typeof(a)) && !iszero(b)
            return -ϕ′0 / 2b
        end

        # cubic
        d = b^2 - 3a*ϕ′0
        if isfinite(d) && d ≥ 0 && !iszero(a)
            α_new = (-b + sqrt(d)) / 3a
            isfinite(α_new) && α_new > 0 && return α_new
        end
    end
    ρ_lo * α
end
