struct WaterModel{T} <: MaterialModel
    K::T
    γ::T
end

function WaterModel(; K::Real, γ::Real)
    WaterModel(K, γ)
end

function update_stress(model::WaterModel, σ::SymmetricSecondOrderTensor{3}, F::SecondOrderTensor{3})
    K = model.K
    γ = model.γ
    p = K * (1/det(F)^γ - 1)
    -p*one(σ)
end
