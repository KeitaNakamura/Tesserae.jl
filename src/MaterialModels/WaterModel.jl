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

function update_stress(model::WaterModel, F::SecondOrderTensor{3})
    B = model.B
    γ = model.γ
    p = B * (1/det(F)^γ - 1)
    -p*one(SymmetricSecondOrderTensor{3, typeof(p)})
end

function deformation_gradient(model::WaterModel, p::Real)
    B = model.B
    γ = model.γ
    detF = (1 / (p/B + 1))^(1/γ)
    (detF)^(1/3) * one(SecondOrderTensor{3, typeof(p)})
end
