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
end

function LinearElastic(; E::T, ν::T) where {T}
    δ = one(SymmetricSecondOrderTensor{3})
    I = one(SymmetricFourthOrderTensor{3})
    λ = (E*ν) / ((1+ν)*(1-2ν))
    μ = E / 2(1 + ν)
    c = λ*δ⊗δ + 2μ*I
    LinearElastic{T}(E, ν, λ, μ, c)
end
