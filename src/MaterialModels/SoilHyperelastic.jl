struct SoilHyperelastic{T} <: MaterialModel
    κ::T
    α::T
    p_ref::T
    μ_ref::T
end

function SoilHyperelastic(; κ::Real, α::Real, p_ref::Real, μ_ref::Real)
    SoilHyperelastic(κ, α, p_ref, μ_ref)
end

function W(model::SoilHyperelastic, ϵᵉ::SymmetricSecondOrderTensor{3})
    κ, α, p_ref, μ_ref = model.κ, model.α, model.p_ref, model.μ_ref
    ϵᵉᵥ = tr(ϵᵉ)
    eᵉ = dev(ϵᵉ)
    Ω = -ϵᵉᵥ/κ + α/κ*(eᵉ ⊡ eᵉ)
    -κ*p_ref*exp(Ω) + μ_ref*(eᵉ ⊡ eᵉ)
end

function ∇W(model::SoilHyperelastic, ϵᵉ::SymmetricSecondOrderTensor{3})
    κ, α, p_ref, μ_ref = model.κ, model.α, model.p_ref, model.μ_ref
    δ = one(ϵᵉ)
    ϵᵉᵥ = tr(ϵᵉ)
    eᵉ = dev(ϵᵉ)
    Ω = -ϵᵉᵥ/κ + α/κ*(eᵉ ⊡ eᵉ)
    p = p_ref * exp(Ω)
    μ = -α*p + μ_ref
    p*δ + 2μ*eᵉ
end

function ∇²W(model::SoilHyperelastic, ϵᵉ::SymmetricSecondOrderTensor{3, T}) where {T}
    κ, α, p_ref, μ_ref = model.κ, model.α, model.p_ref, model.μ_ref
    δ = one(ϵᵉ)
    D = one(SymmetricFourthOrderTensor{3, T}) - 1/3 * δ ⊗ δ
    ϵᵉᵥ = tr(ϵᵉ)
    eᵉ = dev(ϵᵉ)
    Ω = -ϵᵉᵥ/κ + α/κ*(eᵉ ⊡ eᵉ)
    p = p_ref * exp(Ω)
    μ = -α*p + μ_ref
    -p/κ*δ⊗δ + 2α*p/κ*(eᵉ ⊗ δ + δ ⊗ eᵉ - 2α*(eᵉ ⊗ eᵉ)) + 2μ*D
end

function W̃(model::SoilHyperelastic, σ::SymmetricSecondOrderTensor{3})
    κ, α, p_ref, μ_ref = model.κ, model.α, model.p_ref, model.μ_ref
    p = mean(σ)
    s = dev(σ)
    μ = -α*p + μ_ref
    -κ*p*(log(p/p_ref) - 1) + (s ⊡ s)/4μ
end

function ∇W̃(model::SoilHyperelastic, σ::SymmetricSecondOrderTensor{3})
    κ, α, p_ref, μ_ref = model.κ, model.α, model.p_ref, model.μ_ref
    δ = one(σ)
    p = mean(σ)
    s = dev(σ)
    μ = -α*p + μ_ref
    -(κ*log(p/p_ref) - α/4μ^2*(s ⊡ s)) * δ/3 + s/2μ
end

function ∇²W̃(model::SoilHyperelastic, σ::SymmetricSecondOrderTensor{3, T}) where {T}
    κ, α, p_ref, μ_ref = model.κ, model.α, model.p_ref, model.μ_ref
    δ = one(σ)
    D = one(SymmetricFourthOrderTensor{3, T}) - 1/3 * δ ⊗ δ
    p = mean(σ)
    s = dev(σ)
    μ = -α*p + μ_ref
    1/9*(-κ/p + α^2/2μ^3*(s ⊡ s))*(δ ⊗ δ) + α/6μ^2*(s ⊗ δ + δ ⊗ s) + D/2μ
end

function matcalc(::Val{:stress}, model::SoilHyperelastic, ϵᵉ::SymmetricSecondOrderTensor{3})
    ∇W(model, ϵᵉ)
end

function matcalc(::Val{:elastic_strain}, model::SoilHyperelastic, σ::SymmetricSecondOrderTensor{3})
    ∇W̃(model, σ)
end

function matcalc(::Val{:stress}, model::SoilHyperelastic, σ::SymmetricSecondOrderTensor{3}, dϵᵉ::SymmetricSecondOrderTensor{3})
    ϵᵉ = ∇W̃(model, σ) + dϵᵉ
    matcalc(Val(:stress), model, ϵᵉ)
end

function matcalc(::Val{:stiffness}, model::SoilHyperelastic, ϵᵉ::SymmetricSecondOrderTensor{3})
    ∇²W(model, ϵᵉ)
end

function matcalc(::Val{:bulk_modulus}, model::SoilHyperelastic, σ::SymmetricSecondOrderTensor{3})
    -mean(σ)/model.κ
end

function matcalc(::Val{:shear_modulus}, model::SoilHyperelastic, σ::SymmetricSecondOrderTensor{3})
    κ, α, μ_ref = model.κ, model.α, model.μ_ref
    e = dev(matcalc(Val(:elastic_strain), model, σ))
    ϵs = sqrt(2/3 * e ⊡ e)
    p = mean(σ)
    μ = -α*p + μ_ref
    μ - 3*α^2*p*ϵs^2/κ
end
