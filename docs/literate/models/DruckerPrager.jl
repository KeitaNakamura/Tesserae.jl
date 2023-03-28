# # Drucker--Prager model
#
# The following Drucker--Prager model is applied for the material model of the sand:
#
# ```math
# f(\bm{\sigma}) = \sqrt{J_2(\bm{\sigma)}} + \alpha I_1(\bm{\sigma}) - \kappa = 0,
# ```
#
# where $J_2$ is the second deviatoric stress invariant, $I_1$ is the first stress invariant,
# and $\alpha$ and $\kappa$ are the material parameters associated with the cohesion $c$ and
# internal friction angle $\phi$ in the Morh--Coulomb model, respectively.
# Under the plane-strain condition, the material parameters are caculated as follows:
#
# ```math
# \alpha = \frac{\tan\phi}{\sqrt{9+12\tan^2\phi}},\quad
# \kappa = \frac{3c}{\sqrt{9+12\tan^2\phi}}.
# ```
#
# For the plastic flow rule, the following non-associative flow rule is employed:
#
# ```math
# g(\bm{\sigma}) = \sqrt{J_2(\bm{\sigma})} + \beta I_1(\bm{\sigma})
# ```
#
# where $\beta$ is derived from the dilatancy angle $\psi$ as
#
# ```math
# \beta = \frac{\tan\psi}{\sqrt{9+12\tan^2\psi}}
# ```

include("LinearElastic.jl")

struct DruckerPrager{T}
    elastic::LinearElastic{T}
    α::T
    κ::T
    β::T
    p_t::T # mean stress for tension limit
end

function DruckerPrager(elastic::LinearElastic{T}; c::T, ϕ::T, ψ::T = ϕ, p_t::T = c/tan(ϕ)) where {T}
    ## assuming plane strain condition
    κ = 3c     / sqrt(9 + 12tan(ϕ)^2)
    α = tan(ϕ) / sqrt(9 + 12tan(ϕ)^2)
    β = tan(ψ) / sqrt(9 + 12tan(ψ)^2)
    DruckerPrager{T}(elastic, α, κ, β, p_t)
end

function yield_function(model::DruckerPrager, σ::SymmetricSecondOrderTensor{3})
    α, κ = model.α, model.κ
    s = dev(σ)
    I₁ = tr(σ)
    J₂ = s ⊡ s / 2
    √J₂ + α*I₁ - κ
end

# We also define `plastic_flow` to compute the gradient of the plastic potential function
# $\partial{g} / \partial\bm{\sigma}$:

function plastic_flow(model::DruckerPrager, σ::SymmetricSecondOrderTensor{3})
    β = model.β
    s = dev(σ)
    I₁ = tr(σ)
    J₂ = s ⊡ s / 2
    if J₂ < eps(typeof(J₂))
        dgdσ = β*one(σ)
    else
        dgdσ = 2s/√J₂ + β*one(σ)
    end
    dgdσ
end

# ## Return mapping
#
# In the integration process of the stress, we use return mapping algorithm.
# The trial elastic stress can be calculated as
#
# ```math
# \bm{\sigma}^{\mathrm{tr}} = \bm{\sigma}^n + \bm{c}^\mathrm{e} : \Delta\bm{\epsilon}
# ```
#
# If the stress is inside of the yield function, i.e., $f^\mathrm{tr} \le 0$, then the updated stress
# $\bm{\sigma}^{n+1}$ is set to the trial elastic stress as
#
# ```math
# \bm{\sigma}^{n+1} = \bm{\sigma}^\mathrm{tr}.
# ```
#
# If the stress is outside of the yield function, the plastic corrector needs to be performed.

function compute_cauchy_stress(model::DruckerPrager, σⁿ::SymmetricSecondOrderTensor{3}, Δϵ::SymmetricSecondOrderTensor{3})
    ## mean stress for tension limit
    p_t = model.p_t

    ## elastic predictor
    cᵉ = model.elastic.c
    σᵗʳ = σⁿ + cᵉ ⊡ Δϵ
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
            α, κ = model.α, model.κ
            p = mean(σ)
            σ = p_t*I + √2*(κ-α*p)*normalize(s)
        end
    end

    σ
end
