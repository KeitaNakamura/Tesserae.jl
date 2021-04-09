struct WaterModel{T} <: MaterialModel
    K::T
    γ::T
end

# https://www.researchgate.net/publication/220789258_Weakly_Compressible_SPH_for_Free_Surface_Flows
# K = 1119 (kPa)
# γ = 7
function WaterModel(; K::Real = 1119e3, γ::Real = 7)
    WaterModel(K, γ)
end

function update_stress(model::WaterModel, σ::SymmetricSecondOrderTensor{3}, F::SecondOrderTensor{3})
    K = model.K
    γ = model.γ
    p = K * (1/det(F)^γ - 1)
    -p*one(σ)
end
