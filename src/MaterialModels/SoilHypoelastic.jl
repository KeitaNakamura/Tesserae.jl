struct SoilHypoelastic{T} <: MaterialModel
    κ::T
    ν::T
    e0::T
    K_p⁻¹::T
    G_p⁻¹::T
    D_p⁻¹::SymmetricFourthOrderTensor{3, T, 36}
end

SoilHypoelastic(; kwargs...) = SoilHypoelastic{Float64}(; kwargs...)

function SoilHypoelastic{T}(; κ::Real, ν::Real, e0::Real) where {T}
    K = (1 + e0) / κ
    G = 3K * (1-2ν) / 2(1+ν)
    λ = 3K*ν / (1+ν)
    δ = one(SymmetricSecondOrderTensor{3, T})
    I = one(SymmetricFourthOrderTensor{3, T})
    D = λ * δ ⊗ δ + 2G * I
    SoilHypoelastic{T}(κ, ν, e0, K, G, D)
end

function convert_type(::Type{T}, model::SoilHypoelastic) where {T}
    SoilHypoelastic(
        convert(T, model.κ),
        convert(T, model.ν),
        convert(T, model.e0),
        convert(T, model.K_p⁻¹),
        convert(T, model.G_p⁻¹),
        convert(SymmetricFourthOrderTensor{3, T}, model.D_p⁻¹),
    )
end

function matcalc(::Val{:stress}, model::SoilHypoelastic, σ::SymmetricSecondOrderTensor{3}, dϵ::SymmetricSecondOrderTensor{3})::typeof(dϵ)
    @_inline_meta
    σ + calc(Val(:stiffness), model, σ) ⊡ dϵ
end

function calc(::Val{:stiffness}, model::SoilHypoelastic, σ::SymmetricSecondOrderTensor{3})
    model.D_p⁻¹ * abs(mean(σ))
end

function matcalc(::Val{:bulk_modulus}, model::SoilHypoelastic, σ::SymmetricSecondOrderTensor{3})
    model.K_p⁻¹ * abs(mean(σ))
end

function matcalc(::Val{:shear_modulus}, model::SoilHypoelastic, σ::SymmetricSecondOrderTensor{3})
    model.G_p⁻¹ * abs(mean(σ))
end
