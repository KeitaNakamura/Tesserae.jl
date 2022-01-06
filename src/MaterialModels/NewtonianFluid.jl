struct NewtonianFluid{T, Model <: WaterModel} <: MaterialModel
    pressure_model::Model
    μ::T
end

NewtonianFluid(; μ::Real, kwargs...) = NewtonianFluid{Float64}(; μ, kwargs...)
NewtonianFluid{T}(; μ::Real, kwargs...) where {T} = NewtonianFluid(MorrisWaterModel{T}(; kwargs...), convert(T, μ))

function convert_type(::Type{T}, model::NewtonianFluid) where {T}
    NewtonianFluid(
        convert_type(T, model.pressure_model),
        convert(T, model.μ),
    )
end

function matcalc(::Val{:pressure}, model::NewtonianFluid, J::Real)
    matcalc(Val(:pressure), model.pressure_model, J)
end

function matcalc(::Val{:stress}, model::NewtonianFluid, ∇v::SecondOrderTensor{3}, J::Real, dt::Real)
    P = matcalc(Val(:pressure), model, J)
    -P*I + 2*model.μ*dev(symmetric(∇v))
end
