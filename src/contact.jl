struct CoulombFriction
    μ::Float64
    ϕ::Float64
    c::Float64
    separation::Bool
end

"""
    CoulombFriction(; parameters...)

Frictional contact using Mohr-Coulomb criterion.

# Parameters
* `μ`: friction coefficient (use `μ` or `ϕ`)
* `ϕ`: friction angle (radian)
* `c`: cohesion (default: `0`)
* `separation`: `true` or `false` (default: `false`).

If `separation` is `true`, continuum body can leave from the boundary surface.
"""
function CoulombFriction(; μ::Union{Real, Nothing} = nothing, ϕ::Union{Real, Nothing} = nothing, c::Real = 0, separation::Bool = false)
    ( isnothing(μ) &&  isnothing(ϕ)) && throw(ArgumentError("both `μ` and `ϕ` are not found"))
    (!isnothing(μ) && !isnothing(ϕ)) && throw(ArgumentError("both `μ` and `ϕ` are used, choose only one parameter"))
    isnothing(ϕ) && (ϕ = atan(μ))
    isnothing(μ) && (μ =  tan(ϕ))
    CoulombFriction(μ, ϕ, c, separation)
end

"""
    CoulombFriction(:sticky)

This is the same as the `CoulombFriction(; μ = Inf, separation = false)`.

---

    CoulombFriction(:slip; separation = false)

This is the same as the `CoulombFriction(; μ = 0, separation = false)`.
"""
function CoulombFriction(cond::Symbol; separation = false)
    cond == :sticky && return CoulombFriction(; μ = Inf, separation = false)
    cond == :slip   && return CoulombFriction(; μ = 0, separation)
    throw(ArgumentError("Use `:sticky` or `:slip` for contact condition"))
end

issticky(cond::CoulombFriction) = isinf(cond.μ) && !cond.separation
isslip(cond::CoulombFriction) = iszero(cond.μ) && iszero(cond.c)

"""
    contacted(::CoulombFriction, q::Vec, n::Vec)

Compute `q` caused by contact.
`q` can be used as velocity (relative velocity) `v` or force `f`.
`n` is the unit vector normal to the surface.

# Examples
```jldoctest
julia> cond = CoulombFriction(:slip, separation = false);

julia> v = Vec(1.0, -1.0); n = Vec(0.0, 1.0);

julia> v + contacted(cond, v, n)
2-element Vec{2, Float64}:
 1.0
 0.0
```
"""
function contacted(cond::CoulombFriction, q::Vec{dim, T}, n::Vec{dim, T})::Vec{dim, T} where {dim, T}
    # `q` can be used as velocity `v` or force `f`
    # use the relative velocity when using `q` as velocity.

    # Sticky: leaving is also not allowed like chewing gum
    # so just apply `q` with opposite direction
    q_sticky = -q
    issticky(cond) && return q_sticky

    # normal component of `q` to surface
    # `q_norm < 0` means the object is leaving from the surface
    # so applying negative `q_norm` should not be allowed when `cond.separation == true`
    q_norm = q_sticky ⋅ n
    q_norm = ifelse(cond.separation, max(0, q_norm), q_norm)
    q_n = q_norm * n

    # Slip
    # don't need to consider tangent component
    isslip(cond) && return q_n

    # Friction
    if q_norm > 0 # friction force should be considered
        q_t = q_sticky - q_n
        μ = T(cond.μ)
        c = T(cond.c)
        return q_n + min(1, (c + μ*norm(q_n))/norm(q_t)) * q_t # put `norm(q_t)` inside of `min` to handle with deviding zero
    end
    q_n
end

function contacted(cond::CoulombFriction, v::Vec, n::Vec)
    contacted(cond, promote(v, n)...)
end
