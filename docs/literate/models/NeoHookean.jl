# # Neo-Hookean model
#
# The first Piola-Kirchhoff stress $\bm{P}$ is expressed as a function of the deformation gradient $\bm{F}$ as follows:
#
# ```math
# \bm{P} = \mu \left( \bm{F} - \bm{F}^{-\top} \right) + \lambda \ln{J} \bm{F}^{-\top},
# ```
#
# where $\mu$ and $\lambda$ are the Lame's constants.
# Using $\bm{\sigma} = \frac{1}{J}\bm{P}\bm{F}^\top$, the Cauchy stress can be written as
#
# ```math
# \bm{\sigma} = \frac{1}{J} \left[ \mu \left( \bm{F}\bm{F}^\top - \bm{I} \right) + \lambda \ln{J} \bm{I} \right].
# ```

struct NeoHookean{T}
    E::T # Young's modulus
    ν::T # Poissons's ratio
    λ::T # Lame's first parameter
    μ::T # shear modulus
end

function NeoHookean(; E::T, ν::T) where {T}
    λ = (E*ν) / ((1+ν)*(1-2ν))
    μ = E / 2(1 + ν)
    NeoHookean{T}(E, ν, λ, μ)
end

function compute_first_piola_kirchhoff_stress(model::NeoHookean, F::SecondOrderTensor{3})
    λ, μ = model.λ, model.μ
    F⁻ᵀ = inv(F)'
    μ*(F - F⁻ᵀ) + λ*log(det(F))*F⁻ᵀ
end

function compute_cauchy_stress(model::NeoHookean, F::SecondOrderTensor{3})
    λ, μ = model.λ, model.μ
    J = det(F)
    b = symmetric(F⋅F', :U)
    1/J * (μ*(b-I) + λ*log(J)*I)
end
