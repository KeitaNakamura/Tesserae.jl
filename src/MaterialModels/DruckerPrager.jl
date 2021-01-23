struct DruckerPrager{T} <: MaterialModel
    elastic::LinearElastic{T}
    κ::T
    α::T
    β::T
end

function DruckerPrager{T}(; κ::Real, α::Real, β::Real = α, kwargs...) where {T}
    elastic = LinearElastic{T}(; kwargs...)
    DruckerPrager{T}(elastic, κ, α, β)
end

# for Mohr-Coulomb criterion
function DruckerPrager{T}(mc_type::Symbol; c::Real, ϕ::Real, ψ::Real = ϕ, kwargs...) where {T}
    ϕ = deg2rad(ϕ)
    ψ = deg2rad(ψ)
    if mc_type == :circumscribed
        κ = 6c*cos(ϕ) / (√3 * (3 - sin(ϕ)))
        α = 2sin(ϕ) / (√3 * (3 - sin(ϕ)))
        β = 2sin(ψ) / (√3 * (3 - sin(ψ)))
    elseif mc_type == :inscribed
        κ = 6c*cos(ϕ) / (√3 * (3 + sin(ϕ)))
        α = 2sin(ϕ) / (√3 * (3 + sin(ϕ)))
        β = 2sin(ψ) / (√3 * (3 + sin(ψ)))
    elseif mc_type == :plane_strain
        κ = 3c / sqrt(9 + 12tan(ϕ)^2)
        α = tan(ϕ) / sqrt(9 + 12tan(ϕ)^2)
        β = tan(ψ) / sqrt(9 + 12tan(ψ)^2)
    else
        throw(ArgumentError("Choose Mohr-Coulomb type from :circumscribed, :inscribed and :plane_strain"))
    end
    DruckerPrager{T}(; κ, α, β, kwargs...)
end

DruckerPrager(args...; kwargs...) = DruckerPrager{Float64}(args...; kwargs...)

function update_stress(model::DruckerPrager, σ::SymmetricSecondOrderTensor{3}, dϵ::SymmetricSecondOrderTensor{3})::typeof(dϵ)
    # compute the stress at the elastic trial state
    De = model.elastic.D
    σ_trial = σ + De ⊡ dϵ
    # compute the yield function at the elastic trial state
    dfdσ, f_trial = gradient(σ_trial -> yield_function(model, σ_trial), σ_trial, :all)
    f_trial ≤ 0.0 && return σ_trial
    # compute the increment of the plastic multiplier
    dgdσ = plastic_flow(model, σ_trial)
    Δγ = f_trial / (dgdσ ⊡ De ⊡ dfdσ)
    # compute the stress
    σ_trial - Δγ * (De ⊡ dgdσ)
end

function yield_function(model::DruckerPrager, σ::SymmetricSecondOrderTensor{3})::eltype(σ)
    κ = model.κ
    α = model.α
    I₁ = tr(σ)
    s = dev(σ)
    J₂ = (s ⊡ s) / 2
    √J₂ - (κ - α*I₁)
end

function plastic_flow(model::DruckerPrager, σ::SymmetricSecondOrderTensor{3})::typeof(σ)
    β = model.β
    s = dev(σ)
    J₂ = (s ⊡ s) / 2
    if J₂ < eps(typeof(J₂))
        dgdσ = β * one(σ)
    else
        dgdσ = s / (2*√J₂) + β * one(σ)
    end
    dgdσ
end
