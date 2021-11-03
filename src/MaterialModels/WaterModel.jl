# Monaghan model
struct MonaghanWaterModel{T} <: MaterialModel
    B::T
    γ::T
end

# https://www.researchgate.net/publication/220789258_Weakly_Compressible_SPH_for_Free_Surface_Flows
# γ = 7
function MonaghanWaterModel(; B::Real, γ::Real = 7)
    T = promote_type(typeof(B), typeof(γ))
    MonaghanWaterModel{T}(B, γ)
end

function matcalc(::Val{:stress}, model::MonaghanWaterModel, J::Real)
    B = model.B
    γ = model.γ
    p = B * (1/J^γ - 1)
    -p*one(SymmetricSecondOrderTensor{3, typeof(p)})
end

function matcalc(::Val{:jacobian}, model::MonaghanWaterModel, p::Real)
    B = model.B
    γ = model.γ
    (1 / (p/B + 1))^(1/γ)
end
