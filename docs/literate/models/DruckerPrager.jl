# # Drucker--Prager model
#
# The following Drucker--Prager model is applied for the material model of the sand:
#
# ```math
# f(\bm{\sigma}) = \| \bm{s} \| - (A - Bp),
# ```
#
# where $\bm{s}$ is the deviatoric stress, $p$ is the mean stress,
# and $A$ and $B$ are the material parameters associated with the cohesion $c$ and
# internal friction angle $\phi$ in the Morh--Coulomb model, respectively.
#
# For the plastic flow rule, the following non-associative flow rule is employed:
#
# ```math
# g(\bm{\sigma}) = \| \bm{s} \| + bp
# ```
#
# where $b$ is derived from the dilatancy angle $\psi$.
#
# ## Plane-strain condition
#
# Under the plane-strain condition, the material parameters are caculated as follows:
#
# ```math
# A = \frac{3\sqrt{2}c}{\sqrt{9+12\tan^2\phi}},\quad
# B = \frac{3\sqrt{2}\tan\phi}{\sqrt{9+12\tan^2\phi}}.
# ```
#
# For the plastic flow rule, $b$ is derived from the dilatancy angle $\psi$ as
#
# ```math
# b = \frac{3\sqrt{2}\tan\psi}{\sqrt{9+12\tan^2\psi}}
# ```
#
# ## Inner and outer edge approximation
#
# Conincidence at the inner/outer edges of the Mohr--Coulomb surface is obtained when
#
# ```math
# A = \frac{2\sqrt{6}c\cos\phi}{3\pm\sin\phi},\quad
# B = \frac{2\sqrt{6}\sin\phi}{3\pm\sin\phi}.
# ```
#
# The parameter $b$ is given by
#
# ```math
# b = \frac{2\sqrt{6}\sin\psi}{3\pm\sin\psi}.
# ```

include("LinearElastic.jl")

struct DruckerPrager{T}
    elastic::LinearElastic{T}
    A::T
    B::T
    b::T
    p_t::T # mean stress for tension limit
end

function DruckerPrager(type::Symbol, elastic::LinearElastic{T}; c::T, ϕ::T, ψ::T = ϕ, p_t::T = c/tan(ϕ)) where {T}
    if type == :plane_strain
        A = 3√2c      / sqrt(9+12tan(ϕ)^2)
        B = 3√2tan(ϕ) / sqrt(9+12tan(ϕ)^2)
        b = 3√2tan(ψ) / sqrt(9+12tan(ψ)^2)
    elseif type == :outer
        A = 2√6c*cos(ϕ) / (3-sin(ϕ))
        B = 2√6sin(ϕ)   / (3-sin(ϕ))
        b = 2√6sin(ψ)   / (3-sin(ψ))
    elseif type == :inner
        A = 2√6c*cos(ϕ) / (3+sin(ϕ))
        B = 2√6sin(ϕ)   / (3+sin(ϕ))
        b = 2√6sin(ψ)   / (3+sin(ψ))
    else
        error("$type is not supported, choose :plane_strain, :outer or :inner")
    end

    DruckerPrager{T}(elastic, A, B, b, p_t)
end

function yield_function(model::DruckerPrager, σ::SymmetricSecondOrderTensor{3})
    A, B = model.A, model.B
    p = mean(σ)
    s = dev(σ)
    norm(s) - (A - B*p)
end

# We also define `plastic_flow` to compute the gradient of the plastic potential function
# $\partial{g} / \partial\bm{\sigma}$:

function plastic_flow(model::DruckerPrager, σ::SymmetricSecondOrderTensor{3, T}) where {T}
    TOL = sqrt(eps(T))
    b = model.b
    s = dev(σ)
    s_norm = norm(s)
    s/s_norm*!(s_norm<TOL) + b/3*I
end

# ## Return mapping
#
# In the integration process of the stress, we use return mapping algorithm.
# The trial elastic stress can be calculated as
#
# ```math
# \bm{\sigma}^{\mathrm{tr}}
# = \bm{c}^\mathrm{e} : \bm{\epsilon}^\mathrm{e,tr}
# = \bm{c}^\mathrm{e} : \left( \bm{\epsilon}^\mathrm{e}_n + \Delta\bm{\epsilon} \right),
# ```
#
# where the constant elastic stiffness is assumed.
#
# If the stress is inside of the yield function, i.e., $f^\mathrm{tr} \le 0$, then the updated stress
# $\bm{\sigma}^{n+1}$ is set to the trial elastic stress as
#
# ```math
# \bm{\sigma}^{n+1} = \bm{\sigma}^\mathrm{tr}.
# ```
#
# If the stress is outside of the yield function, the plastic corrector needs to be performed.

function compute_stress(model::DruckerPrager, ϵᵉᵗʳ::SymmetricSecondOrderTensor{3})
    ## mean stress for tension limit
    p_t = model.p_t

    ## elastic predictor
    cᵉ, σᵗʳ = gradient(ϵᵉ->compute_stress(model.elastic, ϵᵉ), ϵᵉᵗʳ, :all)
    dfdσ, fᵗʳ = gradient(σ->yield_function(model, σ), σᵗʳ, :all)
    if fᵗʳ ≤ 0 && mean(σᵗʳ) ≤ p_t
        σ = σᵗʳ
        return σ
    end

    ## plastic corrector
    dgdσ = plastic_flow(model, σᵗʳ)
    Δλ = fᵗʳ / (dfdσ ⊡ cᵉ ⊡ dgdσ)
    Δϵᵖ = Δλ * dgdσ
    σ = σᵗʳ - cᵉ ⊡ Δϵᵖ

    ## simple tension cutoff
    if !(mean(σ) < p_t)
        s = dev(σᵗʳ)
        σ = p_t*I + s
        if yield_function(model, σ) > 0
            ## map to corner
            A, B = model.A, model.B
            p = mean(σ)
            σ = p_t*I + (A-B*p)*normalize(s)
        end
    end

    σ
end
