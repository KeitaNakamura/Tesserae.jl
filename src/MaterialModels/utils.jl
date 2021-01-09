@inline volumetric_stress(σ::SymmetricTensor{2, 3}) = mean(σ)
@inline deviatoric_stress(σ::SymmetricTensor{2, 3}) = (s = dev(σ); sqrt(3/2 * s ⊡ s))
@inline volumetric_strain(ϵ::SymmetricTensor{2, 3}) = tr(ϵ)
@inline deviatoric_strain(ϵ::SymmetricTensor{2, 3}) = (e = dev(ϵ); sqrt(2/3 * e ⊡ e))
@inline infinitesimal_strain(F::Tensor{2, 3}) = symmetric(F - one(F))

function soundspeed(K::Real, G::Real, ρ::Real)
    sqrt((K + 4G/3) / ρ)
end

function jaumann_stress(σ::SymmetricTensor{2, 3}, σ_n::SymmetricTensor{2, 3}, L::Tensor{2, 3}, dt::Real)
    W = skew(L)
    σ + dt * symmetric(W ⋅ σ_n - σ_n ⋅ W)
end

#=
function green_naghdi_stress(σ::SymmetricTensor{2, 3}, σ_n::SymmetricTensor{2, 3}, F::Tensor{2, 3}, dt::Real)
    σ + dt * symmetric(σ_n ⋅ dR ⋅ R' - dR ⋅ R' ⋅ σ_n)
end

function polar_right(F::SecondOrderTensor)
    U² = tdot(F)
    U = _eigen_back_sqrt(U²)
    symmetric(U), F ⋅ inv(U)
end

function polar_left(F::SecondOrderTensor)
    V² = dott(F)
    V = _eigen_back_sqrt(U²)
    symmetric(U), inv(V) ⋅ F
end

function _eigen_back_sqrt(A::SymmetricTensor{2, dim}) where {dim}
    E = eigen(A)
    sum(ntuple(Val(dim)) do d
        λ_d = sqrt(E.values[d])
        n = Vec{dim}(E.vectors[:,d])
        λ_d * (n ⊗ n)
    end)
end
=#
