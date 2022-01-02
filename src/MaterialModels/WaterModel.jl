abstract type WaterModel <: MaterialModel end

# Monaghan model
struct MonaghanWaterModel{T} <: WaterModel
    B::T
    γ::T
end

# https://www.researchgate.net/publication/220789258_Weakly_Compressible_SPH_for_Free_Surface_Flows
# γ = 7
MonaghanWaterModel(; kwargs...) = MonaghanWaterModel{Float64}(; kwargs...)
function MonaghanWaterModel{T}(; B::Real, γ::Real = 7) where {T}
    MonaghanWaterModel{T}(B, γ)
end

function convert_type(::Type{T}, model::MonaghanWaterModel) where {T}
    MonaghanWaterModel{T}(
        convert(T, model.B),
        convert(T, model.γ),
    )
end

function matcalc(::Val{:pressure}, model::MonaghanWaterModel, J::Real)
    B = model.B
    γ = model.γ
    B * (1/J^γ - 1)
end

function matcalc(::Val{:jacobian}, model::MonaghanWaterModel, p::Real)
    B = model.B
    γ = model.γ
    (1 / (p/B + 1))^(1/γ)
end


struct SimpleWaterModel{T} <: WaterModel
    ρ0::T
    P0::T
    c::T # speed of sound
end

SimpleWaterModel(; kwargs...) = SimpleWaterModel{Float64}(; kwargs...)
function SimpleWaterModel{T}(; ρ0::Real, P0::Real, c::Real) where {T}
    SimpleWaterModel{T}(ρ0, P0, c)
end

function convert_type(::Type{T}, model::SimpleWaterModel) where {T}
    SimpleWaterModel(
        convert(T, model.ρ0),
        convert(T, model.P0),
        convert(T, model.c),
    )
end

function matcalc(::Val{:pressure}, model::SimpleWaterModel, J::Real)
    ρ0, P0, c = model.ρ0, model.P0, model.c
    ρ = ρ0 / J
    P0 + c^2*(ρ - ρ0)
end
