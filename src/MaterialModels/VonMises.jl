struct VonMises{T} <: MaterialModel
    elastic::LinearElastic{T}
    q_y::T
end

function VonMises(elastic::LinearElastic; q_y::Real)
    VonMises(elastic, q_y)
end

function VonMises(elastic::LinearElastic, model_type; c::Real)
    model_type = Symbol(model_type)
    if model_type == :plane_strain
        q_y = √3c
    else
        throw(ArgumentError("Supported model type is :plane_strain, got $model_type"))
    end
    VonMises(elastic, q_y)
end

function matcalc(::Val{:stress}, model::VonMises, σ::SymmetricSecondOrderTensor{3}, dϵ::SymmetricSecondOrderTensor{3})::typeof(dϵ)
    # compute the stress at the elastic trial state
    De = model.elastic.D
    σ_trial = σ + De ⊡ dϵ
    # compute the yield function at the elastic trial state
    dfdσ, f_trial = gradient(σ_trial -> matcalc(Val(:yield_function), model, σ_trial), σ_trial, :all)
    f_trial ≤ 0.0 && return σ_trial
    # compute the increment of the plastic multiplier
    dgdσ = matcalc(Val(:plastic_flow), model, σ_trial)
    Δγ = f_trial / (dgdσ ⊡ De ⊡ dfdσ)
    # compute the stress
    σ_trial - Δγ * (De ⊡ dgdσ)
end

function matcalc(::Val{:yield_function}, model::VonMises, σ::SymmetricSecondOrderTensor{3})::eltype(σ)
    s = dev(σ)
    q = sqrt(3/2 * s ⊡ s)
    q - model.q_y
end

function matcalc(::Val{:plastic_flow}, model::VonMises, σ::SymmetricSecondOrderTensor{3})::typeof(σ)
    s = dev(σ)
    _s_ = sqrt(s ⊡ s)
    if _s_ < √eps(eltype(σ))
        dgdσ = zero(s)
    else
        dgdσ = sqrt(3/2) * s / _s_
    end
    dgdσ
end
