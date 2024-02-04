"""
    WLS(::Kernel)

WLS (weighted least squares) interpolation using the linear polynomial.

This interpolation function is used in the moving least squares material point method (MLS-MPM) [^MLSMPM],
but it is referred as the WLS in this library because the formulation is fundamentally WLS scheme.

[^MLSMPM]: [Hu, Y., Fang, Y., Ge, Z., Qu, Z., Zhu, Y., Pradhana, A., & Jiang, C. (2018). A moving least squares material point method with displacement discontinuity and two-way rigid body coupling. *ACM Transactions on Graphics (TOG)*, 37(4), 1-14.](https://doi.org/10.1145/3197517.3201293)
"""
struct WLS{K <: Kernel} <: Interpolation
    kernel::K
end

get_kernel(wls::WLS) = wls.kernel
gridspan(wls::WLS) = gridspan(get_kernel(wls))
@inline neighbornodes(wls::WLS, pt, lattice::Lattice) = neighbornodes(get_kernel(wls), pt, lattice)

# general version
function update_property!(mp::MPValues{<: WLS}, pt, lattice::Lattice{dim, T}, filter::AbstractArray{Bool} = Trues(size(lattice))) where {dim, T}
    indices = neighbornodes(mp)

    it = interpolation(mp)
    F = get_kernel(it)
    M = zero(Mat{dim+1, dim+1, T})
    xₚ = getx(pt)

    @inbounds @simd for ip in eachindex(indices)
        i = indices[ip]
        xᵢ = lattice[i]
        w = value(F, pt, lattice, i) * filter[i]
        p = [1; xᵢ-xₚ]
        M += w * p ⊗ p
        mp.N[ip] = w
    end

    M⁻¹ = inv(M)

    @inbounds @simd for ip in eachindex(indices)
        i = indices[ip]
        xᵢ = lattice[i]
        wq = mp.N[ip] * (M⁻¹ ⋅ [1;xᵢ-xₚ])
        mp.N[ip] = wq[1]
        mp.∇N[ip] = @Tensor wq[2:end]
    end
end

# fast version for `LinearWLS(BSpline{order}())`
function update_property!(mp::MPValues{<: WLS{<: BSpline}}, pt, lattice::Lattice, filter::AbstractArray{Bool} = Trues(size(lattice)))
    indices = neighbornodes(mp)
    isnearbounds = size(mp.N) != size(indices) || !alltrue(filter, indices)
    if isnearbounds
        fast_update_property_nearbounds!(mp, pt, lattice, filter)
    else
        fast_update_property!(mp, pt, lattice)
    end
end

function fast_update_property!(mp::MPValues{<: WLS{<: BSpline}}, pt, lattice::Lattice{dim, T}) where {dim, T}
    indices = neighbornodes(mp)
    F = get_kernel(interpolation(mp))
    xₚ = getx(pt)
    D = zero(Vec{dim, T}) # diagonal entries of M
    values_gradients!(mp.N, mp.∇N, F, xₚ, lattice)

    @inbounds @simd for ip in eachindex(indices)
        i = indices[ip]
        xᵢ = lattice[i]
        D += w * (xᵢ - xₚ) .* (xᵢ - xₚ)
        mp.∇N[ip] = w * (xᵢ - xₚ)
    end

    D⁻¹ = inv.(D)
    broadcast!(.*, mp.∇N, mp.∇N, D⁻¹)
end

function fast_update_property_nearbounds!(mp::MPValues{<: WLS{<: BSpline}}, pt, lattice::Lattice{dim, T}, filter::AbstractArray{Bool}) where {dim, T}
    indices = neighbornodes(mp)
    it = interpolation(mp)
    F = get_kernel(it)
    P = get_basis(it)
    M = zero(Mat{dim+1, dim+1, T})
    xₚ = getx(pt)

    @inbounds @simd for ip in eachindex(indices)
        i = indices[ip]
        xᵢ = lattice[i]
        w = value(F, xₚ, lattice, i) * filter[i]
        p = value(P, xᵢ - xₚ)
        M += w * p ⊗ p
        mp.N[ip] = w
    end

    M⁻¹ = inv(M)

    @inbounds @simd for ip in eachindex(indices)
        i = indices[ip]
        xᵢ = lattice[i]
        wq = mp.N[ip] * (M⁻¹ ⋅ [1;xᵢ-xₚ])
        mp.N[ip] = wq[1]
        mp.∇N[ip] = @Tensor wq[2:end]
    end
end

Base.show(io::IO, wls::WLS) = print(io, WLS, "(", get_kernel(wls), ")")
