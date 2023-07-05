abstract type LinearSolver end
abstract type NonlinearSolver end

mutable struct GMRESSolver{T} <: LinearSolver
    maxiter::Int
    abstol::T
    reltol::T
    # for adaptive linear parameter
    adaptive::Bool
    counter::Int
    reltol′::T
end

function GMRESSolver(::Type{T}=Float64; maxiter::Int=20, abstol::Real=sqrt(eps(T)), reltol::Real=sqrt(eps(T)), adaptive::Bool=false) where {T}
    GMRESSolver{T}(maxiter, abstol, reltol, adaptive, 0, reltol)
end

function solve!(x, A, b, solver::GMRESSolver)
    if solver.adaptive
        gmres!(fillzero!(x), A, b; solver.maxiter, initially_zero=true, solver.abstol, reltol=solver.reltol′)
    else
        gmres!(fillzero!(x), A, b; solver.maxiter, initially_zero=true, solver.abstol, solver.reltol)
    end
end

function solve!(v::AbstractVector, residual_jacobian!, R::AbstractVector, J, solver::NonlinearSolver, gmres::GMRESSolver)
    if gmres.adaptive
        solve_adaptive!(v, residual_jacobian!, R, J, solver, gmres)
    else
        solve!(v, residual_jacobian!, R, J, solver, (x,A,b)->solve!(x,A,b,gmres))
    end
end

function solve_adaptive!(v::AbstractVector, residual_jacobian!, R::AbstractVector, J, solver::NonlinearSolver, gmres::GMRESSolver{T}) where {T}
    v⁰ = copy(v)

    if gmres.counter ≥ 100 && gmres.reltol′ < 1e-4
        gmres.reltol′ *= 10
        gmres.counter = 0
    end

    k = 1
    converged = false
    while !converged && gmres.reltol′ > eps(T)
        converged = solve!(v, residual_jacobian!, R, J, solver, (x,A,b)->solve!(x,A,b,gmres))
        if converged
            # if converged at the first iteration, +1 counter
            gmres.counter = ifelse(k==1, gmres.counter+1, 0)
            break
        end

        # if not converged, reset the state
        v .= v⁰

        # also make tolerance smaller
        gmres.counter = 0
        gmres.reltol′ /= 10

        k += 1
    end

    converged
end

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
    r0 = zero(eltype(R))
    @inbounds for k in 1:maxiter
        RJ!(R, J, x)
        r = norm(R)
        k == 1 && (r0 = r)
        (r < abstol || r/r0 < reltol) && return true
        linsolve(δx, J, rmul!(R, -1))
        @. x += δx
    end
    false
end
