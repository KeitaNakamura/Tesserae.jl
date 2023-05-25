struct WLS{B <: AbstractBasis, K <: Kernel} <: Interpolation
end

"""
    LinearWLS(::Kernel)

WLS (weighted least squares) interpolation using the linear polynomial.

This interpolation function is used in the moving least squares material point method (MLS-MPM) [^MLSMPM],
but it is referred as the WLS in Marble.jl because the formulation is fundamentally WLS scheme.
For the MLS-MPM formulation, use this `LinearWLS` with [`WLSTransfer`](@ref).

[^MLSMPM]: [Hu, Y., Fang, Y., Ge, Z., Qu, Z., Zhu, Y., Pradhana, A., & Jiang, C. (2018). A moving least squares material point method with displacement discontinuity and two-way rigid body coupling. *ACM Transactions on Graphics (TOG)*, 37(4), 1-14.](https://doi.org/10.1145/3197517.3201293)
"""
const LinearWLS = WLS{PolynomialBasis{1}}

const BilinearWLS = WLS{BilinearBasis}

WLS{B}(w::Kernel) where {B} = WLS{B, typeof(w)}()

get_basis(::WLS{B}) where {B} = B()
get_kernel(::WLS{B, W}) where {B, W} = W()
gridsize(wls::WLS) = gridsize(get_kernel(wls))
@inline neighbornodes(wls::WLS, lattice::Lattice, pt) = neighbornodes(get_kernel(wls), lattice, pt)

function MPValuesInfo{dim, T}(itp::WLS) where {dim, T}
    dims = nfill(gridsize(itp), Val(dim))
    L = length(value(get_basis(itp), zero(Vec{dim, T})))
    values = (; w=zero(T), N=zero(T), ∇N=zero(Vec{dim, T}), Minv=zero(Mat{L, L, T}))
    sizes = (dims, dims, dims, (1,))
    MPValuesInfo{dim, T}(values, sizes)
end

# general version
function update_mpvalues!(mp::SubMPValues, itp::WLS, lattice::Lattice, spy::AbstractArray{Bool}, pt)
    indices = neighbornodes(mp)

    F = get_kernel(itp)
    P = get_basis(itp)
    M = zero(mp.Minv[])
    xₚ = getx(pt)

    @inbounds @simd for j in CartesianIndices(indices)
        i = indices[j]
        xᵢ = lattice[i]
        w = value(F, lattice, i, pt) * spy[i]
        p = value(P, xᵢ - xₚ)
        M += w * p ⊗ p
        mp.w[j] = w
    end

    M⁻¹ = inv(M)

    @inbounds @simd for j in CartesianIndices(indices)
        i = indices[j]
        xᵢ = lattice[i]
        q = M⁻¹ ⋅ value(P, xᵢ - xₚ)
        wq = mp.w[j] * q
        mp.N[j] = wq ⋅ value(P, xₚ - xₚ)
        mp.∇N[j] = wq ⋅ gradient(P, xₚ - xₚ)
    end
    mp.Minv[] = M⁻¹
end

# fast version for `LinearWLS(BSpline{order}())`
function update_mpvalues!(mp::SubMPValues, itp::WLS{PolynomialBasis{1}, <: BSpline}, lattice::Lattice, spy::AbstractArray{Bool}, pt)
    if isnearbounds(mp)
        fast_update_mpvalues_nearbounds!(mp, itp, lattice, spy, pt)
    else
        fast_update_mpvalues!(mp, itp, lattice, spy, pt)
    end
end

function fast_update_mpvalues!(mp::SubMPValues{dim, T}, itp::WLS, lattice::Lattice, spy::AbstractArray{Bool}, pt) where {dim, T}
    indices = neighbornodes(mp)
    F = get_kernel(itp)
    xₚ = getx(pt)
    D = zero(Vec{dim, T}) # diagonal entries
    values_gradients!(mp.w, reinterpret(reshape, T, mp.∇N), F, lattice, xₚ)

    @inbounds @simd for j in CartesianIndices(indices)
        i = indices[j]
        xᵢ = lattice[i]
        w = mp.w[j]
        D += w * (xᵢ - xₚ) .* (xᵢ - xₚ)
        mp.N[j] = w
        mp.∇N[j] = w * (xᵢ - xₚ)
    end

    D⁻¹ = inv.(D)
    broadcast!(.*, mp.∇N, mp.∇N, D⁻¹)
    mp.Minv[] = diagm(vcat(1, D⁻¹))
end

function fast_update_mpvalues_nearbounds!(mp::SubMPValues, itp::WLS, lattice::Lattice, spy::AbstractArray{Bool}, pt)
    indices = neighbornodes(mp)
    F = get_kernel(itp)
    P = get_basis(itp)
    xₚ = getx(pt)
    M = zero(mp.Minv[])

    @inbounds @simd for j in CartesianIndices(indices)
        i = indices[j]
        xᵢ = lattice[i]
        w = value(F, lattice, i, xₚ) * spy[i]
        p = value(P, xᵢ - xₚ)
        M += w * p ⊗ p
        mp.w[j] = w
    end

    M⁻¹ = inv(M)

    @inbounds @simd for j in CartesianIndices(indices)
        i = indices[j]
        xᵢ = lattice[i]
        q = M⁻¹ ⋅ value(P, xᵢ - xₚ)
        wq = mp.w[j] * q
        mp.N[j] = wq[1]
        mp.∇N[j] = @Tensor wq[2:end]
    end
    mp.Minv[] = M⁻¹

    mp
end

Base.show(io::IO, wls::WLS{B}) where {B} = print(io, WLS{B}, "(", get_kernel(wls), ")")
