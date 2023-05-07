# # Linear elastic model
#
# ```math
# \begin{align*}
# \sigma_{ij} &= c_{ijkl} \epsilon_{kl} \\
# c_{ijkl} &= \lambda\delta_{ij}\delta_{kl} + \mu (\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk})
# \end{align*}
# ```

struct LinearElastic{T}
    E::T # Young's modulus
    ν::T # Poissons's ratio
    λ::T # Lame's first parameter
    μ::T # shear modulus
    c::SymmetricFourthOrderTensor{3, T, 36}
    c_inv::SymmetricFourthOrderTensor{3, T, 36}
    c33::Mat{3, 3, T, 9}
    c33_inv::Mat{3, 3, T, 9}
end

function LinearElastic(; E::T, ν::T) where {T}
    λ = (E*ν) / ((1+ν)*(1-2ν))
    μ = E / 2(1 + ν)

    δ = one(SymmetricSecondOrderTensor{3})
    I = one(SymmetricFourthOrderTensor{3})
    c = λ*δ⊗δ + 2μ*I

    δ = ones(Vec{3})
    I = one(Mat{3,3})
    c33 = λ*δ⊗δ + 2μ*I
    LinearElastic{T}(E, ν, λ, μ, c, inv(c), c33, inv(c33))
end

compute_stress(model::LinearElastic, ϵ::SymmetricSecondOrderTensor{3}) = model.c ⊡ ϵ
compute_stress(model::LinearElastic, ϵ::Vec{3}) = model.c33 ⋅ ϵ

compute_strain(model::LinearElastic, σ::SymmetricSecondOrderTensor{3}) = model.c_inv ⊡ σ
compute_strain(model::LinearElastic, σ::Vec{3}) = model.c33_inv ⋅ σ
