struct LinearElastic{T} <: MaterialModel
    E::T
    K::T
    G::T
    λ::T
    ν::T
    D::SymmetricFourthOrderTensor{3, T, 36}
    Dinv::SymmetricFourthOrderTensor{3, T, 36}
end

function LinearElastic(; kwargs...)
    params = kwargs.data
    if haskey(params, :K)
        K = params.K
        if haskey(params, :E)
            E = params.E
            λ = 3K*(3K-E) / (9K-E)
            G = 3K*E / (9K-E)
            ν = (3K-E) / 6K
        elseif haskey(params, :λ)
            λ = params.λ
            E = 9K*(K-λ) / (3K-λ)
            G = 3(K-λ) / 2
            ν = λ / (3K-λ)
        elseif haskey(params, :G)
            G = params.G
            E = 9K*G / (3K+G)
            λ = K - 2G/3
            ν = (3K-2G) / 2(3K+G)
        elseif haskey(params, :ν)
            ν = params.ν
            E = 3K*(1-2ν)
            λ = 3K*ν / (1+ν)
            G = 3K*(1-2ν) / 2(1+ν)
        end
    elseif haskey(params, :E)
        E = params.E
        if haskey(params, :λ)
            λ = params.λ
            R = √(E^2 + 9λ^2 + 2E*λ)
            K = (E+3λ+R) / 6
            G = (E-3λ+R) / 4
            ν = 2λ / (E+λ+R)
        elseif haskey(params, :G)
            G = params.G
            K = E*G / 3(3G-E)
            λ = G*(E-2G) / (3G-E)
            ν = E/2G - 1
        elseif haskey(params, :ν)
            ν = params.ν
            K = E / 3(1-2ν)
            λ = E*ν / ((1+ν)*(1-2ν))
            G = E / 2(1+ν)
        end
    elseif haskey(params, :λ)
        λ = params.λ
        if haskey(params, :G)
            G = params.G
            K = λ + 2G/3
            E = G*(3λ+2G) / (λ+G)
            ν = λ / 2(λ+G)
        elseif haskey(params, :ν)
            ν = params.ν
            K = λ*(1+ν) / 3ν
            E = λ*(1+ν)*(1-2ν) / ν
            G = λ*(1-2ν) / 2ν
        end
    elseif haskey(params, :G)
        G = params.G
        if haskey(params, :ν)
            ν = params.ν
            K = 2G*(1+ν) / 3(1-2ν)
            E = 2G*(1+ν)
            λ = 2G*ν / (1-2ν)
        end
    end
    T = promote_type(typeof.((G, ν, K, E, λ))...)
    δ = one(SymmetricSecondOrderTensor{3, T})
    I = one(SymmetricFourthOrderTensor{3, T})
    D = λ * δ ⊗ δ + 2G * I
    LinearElastic(E, K, G, λ, ν, D, inv(D))
end

function update_stress(model::LinearElastic, σ::SymmetricSecondOrderTensor{3}, dϵ::SymmetricSecondOrderTensor{3})::typeof(dϵ)
    @_inline_meta
    σ + model.D ⊡ dϵ
end
