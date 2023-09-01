using IterativeSolvers

abstract type NonlinearSolver end

struct NewtonSolver{T} <: NonlinearSolver
    maxiter::Int
    abstol::T
    reltol::T
end

function NewtonSolver(::Type{T}=Float64; maxiter::Integer=20, abstol::Real=sqrt(eps(T)), reltol::Real=sqrt(eps(T))) where {T}
    NewtonSolver{T}(maxiter, abstol, reltol)
end

function solve!(x::AbstractVector, residual_jacobian!, R::AbstractVector, J, solver::NewtonSolver, linsolve!::Function=(x,A,b)->x.=A\b)
    newton!(x, residual_jacobian!, R, J; solver.maxiter, solver.abstol, solver.reltol, linsolve = (x,A,b) -> linsolve!(x,A,b))
end

function newton!(x::AbstractVector, RJ!, R::AbstractVector{T}, J, δx::AbstractVector=similar(x);
                 maxiter::Int=20, abstol::T=sqrt(eps(T)), reltol::T=sqrt(eps(T)), linsolve=(x,A,b)->x.=A\b) where {T}
    RJ!(R, J, x)
    r0 = r = norm(R)
    r < abstol && return true
    r_prev = r
    x_prev = copy(x)
    @inbounds for _ in 1:maxiter
        α₀ = α₁ = one(T)
        r_α₀ = r_α₁ = r
        linsolve(δx, J, rmul!(R, -1))
        for k in 1:100
            @. x = x_prev + α₁ * δx
            RJ!(R, J, x)
            r_α₁ = norm(R)
            r_α₁ < r && (r=r_α₁; break)

            if k == 1
                α₂ = linesearch_quadratic_interpolation(α₁, r, -r, r_α₁)
            else
                α₂ = linesearch_cubic_interpolation(α₀, α₁, r, -r, r_α₀, r_α₁)
            end
            # α₂ = ifelse(0.1α₁ < α₂ < 0.5α₁, α₂, α₁/2)
            α₂ = clamp(α₂, 0.1α₁, 0.5α₁)

            α₀ = α₁
            α₁ = α₂
            r_α₀ = r_α₁
        end
        dr = abs(r-r_prev)
        (r < abstol || r/r0 < reltol || dr < abstol || dr/r0 < reltol) && return true

        r_prev = r
        x_prev .= x
    end

    false
end

function linesearch_quadratic_interpolation(α₀, ϕ_0, ϕ′_0, ϕ_α₀)
    -(ϕ′_0 * α₀^2) / 2(ϕ_α₀ - ϕ_0 - ϕ′_0 * α₀)
end

function linesearch_cubic_interpolation(α₀, α₁, ϕ_0, ϕ′_0, ϕ_α₀, ϕ_α₁)
    m = @Mat[α₀^2 -α₁^2
             -α₀^3 α₁^3]
    v = @Vec[ϕ_α₁ - ϕ_0 - ϕ′_0 * α₁
             ϕ_α₀ - ϕ_0 - ϕ′_0 * α₀]
    a, b = (m ⋅ v) / (α₀^2 * α₁^2 * (α₁ - α₀))
    (-b + √(b^2 - 3*a*ϕ′_0)) / 3a
end
