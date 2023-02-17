struct WLS{B <: AbstractBasis, K <: Kernel} <: Interpolation
end

const LinearWLS = WLS{PolynomialBasis{1}}
const BilinearWLS = WLS{BilinearBasis}

WLS{B}(w::Kernel) where {B} = WLS{B, typeof(w)}()

get_basis(::WLS{B}) where {B} = B()
get_kernel(::WLS{B, W}) where {B, W} = W()
@inline neighbornodes(wls::WLS, lattice::Lattice, pt) = neighbornodes(get_kernel(wls), lattice, pt)

mutable struct WLSValue{dim, T, B, K, L, L²} <: MPValue{dim, T}
    itp::WLS{B, K}
    w::Vector{T}
    N::Vector{T}
    ∇N::Vector{Vec{dim, T}}
    Minv::Mat{L, L, T, L²}
end

function MPValue{dim, T}(itp::WLS{B, K}) where {dim, T, B, K}
    L = length(value(get_basis(itp), zero(Vec{dim, T})))
    w = Vector{T}(undef, 0)
    N = Vector{T}(undef, 0)
    ∇N = Vector{Vec{dim, T}}(undef, 0)
    Minv = zero(Mat{L, L, T})
    WLSValue(itp, w, N, ∇N, Minv)
end

num_nodes(mp::WLSValue) = length(mp.N)
get_basis(mp::WLSValue) = get_basis(mp.itp)
@inline shape_value(mp::WLSValue, j::Int) = (@_propagate_inbounds_meta; mp.N[j])
@inline shape_gradient(mp::WLSValue, j::Int) = (@_propagate_inbounds_meta; mp.∇N[j])

# general version
function update_mpvalue!(mp::WLSValue, lattice::Lattice, sppat::AbstractArray{Bool}, pt)
    indices, _ = neighbornodes(mp.itp, lattice, pt)

    n = length(indices)
    resize!(mp.N, n)
    resize!(mp.∇N, n)
    resize!(mp.w, n)

    F = get_kernel(mp.itp)
    P = get_basis(mp.itp)
    M = zero(mp.Minv)
    xp = getx(pt)

    @inbounds for (j, i) in enumerate(indices)
        xi = lattice[i]
        w = value(F, lattice, i, pt) * sppat[i]
        p = value(P, xi - xp)
        M += w * p ⊗ p
        mp.w[j] = w
    end

    Minv = inv(M)

    @inbounds for (j, i) in enumerate(indices)
        xi = lattice[i]
        q = Minv ⋅ value(P, xi - xp)
        wq = mp.w[j] * q
        mp.N[j] = wq ⋅ value(P, xp - xp)
        mp.∇N[j] = wq ⋅ gradient(P, xp - xp)
    end
    mp.Minv = Minv

    indices
end

# fast version for `LinearWLS(BSpline{order}())`
function update_mpvalue!(mp::WLSValue{<: Any, <: Any, PolynomialBasis{1}, <: BSpline}, lattice::Lattice, sppat::AbstractArray{Bool}, pt)
    indices, isnearbounds = neighbornodes(mp.itp, lattice, pt)

    n = length(indices)
    resize!(mp.N, n)
    resize!(mp.∇N, n)
    resize!(mp.w, n)

    if !isnearbounds && @inbounds alltrue(sppat, indices)
        fast_update_mpvalue!(mp, lattice, sppat, indices, pt)
    else
        fast_update_mpvalue_nearbounds!(mp, lattice, sppat, indices, pt)
    end

    indices
end

function fast_update_mpvalue!(mp::WLSValue{dim, T}, lattice::Lattice, sppat::AbstractArray{Bool}, indices, pt) where {dim, T}
    F = get_kernel(mp.itp)
    xp = getx(pt)
    D = zero(Vec{dim, T}) # diagonal entries
    wᵢ = first(values_gradients(F, lattice, xp))

    @inbounds for (j, i) in enumerate(indices)
        xi = lattice[i]
        w = wᵢ[j]
        D += w * (xi - xp) .* (xi - xp)
        mp.w[j] = w
        mp.N[j] = w
        mp.∇N[j] = w * (xi - xp)
    end

    D⁻¹ = inv.(D)
    broadcast!(.*, mp.∇N, mp.∇N, D⁻¹)
    mp.Minv = diagm(vcat(1, D⁻¹))
end

function fast_update_mpvalue_nearbounds!(mp::WLSValue, lattice::Lattice, sppat::AbstractArray{Bool}, indices, pt)
    F = get_kernel(mp.itp)
    P = get_basis(mp.itp)
    xp = getx(pt)
    M = zero(mp.Minv)

    @inbounds for (j, i) in enumerate(indices)
        xi = lattice[i]
        w = value(F, lattice, i, xp) * sppat[i]
        p = value(P, xi - xp)
        M += w * p ⊗ p
        mp.w[j] = w
    end

    Minv = inv(M)

    @inbounds for (j, i) in enumerate(indices)
        xi = lattice[i]
        q = Minv ⋅ value(P, xi - xp)
        wq = mp.w[j] * q
        mp.N[j] = wq[1]
        mp.∇N[j] = @Tensor wq[2:end]
    end
    mp.Minv = Minv

    mp
end
