struct WaterModel{T} <: MaterialModel
    K::T
    γ::T
end

function WaterModel{T}(; K::Real, γ::Real) where {T}
    WaterModel{T}(K, γ)
end
WaterModel(; kwargs...) = WaterModel{Float64}(; kwargs...)

function update_stress(model::WaterModel, σ::SymmetricTensor{2, 3}, F::Tensor{2, 3})::typeof(σ)
    K = model.K
    γ = model.γ
    p = K * (1/det(F)^γ - 1)
    return -p*one(σ)
end
