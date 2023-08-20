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
