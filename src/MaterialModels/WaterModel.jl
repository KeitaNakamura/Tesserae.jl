# Monaghan model
struct WaterModel{T} <: MaterialModel
    B::T
    γ::T
end

# https://www.researchgate.net/publication/220789258_Weakly_Compressible_SPH_for_Free_Surface_Flows
# γ = 7
function WaterModel(; B::Real, γ::Real = 7)
    T = promote_type(typeof(B), typeof(γ))
    WaterModel{T}(B, γ)
end

function update_stress(model::WaterModel, σ::SymmetricSecondOrderTensor{3}, F::SecondOrderTensor{3})
    B = model.B
    γ = model.γ
    p = B * (1/det(F)^γ - 1)
    -p*one(σ)
end
