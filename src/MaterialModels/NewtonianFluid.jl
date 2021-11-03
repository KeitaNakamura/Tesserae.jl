struct NewtonianFluid{T}
    μ::T
end

function NewtonianFluid(; μ::Real)
    NewtonianFluid(μ)
end

function matcalc(::Val{:stress}, model::NewtonianFluid, p::Real, ϵ̇::SymmetricSecondOrderTensor{3})
    -p*one(SymmetricSecondOrderTensor{3, typeof(p)}) + 2*model.μ*dev(ϵ̇)
end
