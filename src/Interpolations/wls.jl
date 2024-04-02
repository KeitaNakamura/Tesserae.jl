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
@inline neighboringnodes(wls::WLS, pt, mesh::CartesianMesh) = neighboringnodes(get_kernel(wls), pt, mesh)

# general version
function update_property!(mp::MPValues{<: WLS}, pt, mesh::CartesianMesh{dim, T}, filter::AbstractArray{Bool} = Trues(size(mesh))) where {dim, T}
    indices = neighboringnodes(mp)

    it = interpolation(mp)
    F = get_kernel(it)
    M = zero(Mat{dim+1, dim+1, T})
    xₚ = getx(pt)

    @inbounds @simd for ip in eachindex(indices)
        i = indices[ip]
        xᵢ = mesh[i]
        w = value(F, pt, mesh, i) * filter[i]
        p = [1; xᵢ-xₚ]
        M += w * p ⊗ p
        mp.N[ip] = w
    end

    M⁻¹ = inv(M)

    @inbounds @simd for ip in eachindex(indices)
        i = indices[ip]
        xᵢ = mesh[i]
        wq = mp.N[ip] * (M⁻¹ ⋅ [1;xᵢ-xₚ])
        mp.N[ip] = wq[1]
        mp.∇N[ip] = @Tensor wq[2:end]
    end
end

# fast version for `LinearWLS(BSpline{order}())`
function update_property!(mp::MPValues{<: WLS{<: BSpline}}, pt, mesh::CartesianMesh, filter::AbstractArray{Bool} = Trues(size(mesh)))
    indices = neighboringnodes(mp)
    isnearbounds = size(mp.N) != size(indices) || !alltrue(filter, indices)
    if isnearbounds
        fast_update_property_nearbounds!(mp, pt, mesh, filter)
    else
        fast_update_property!(mp, pt, mesh)
    end
end

function fast_update_property!(mp::MPValues{<: WLS{<: BSpline}}, pt, mesh::CartesianMesh{dim, T}) where {dim, T}
    indices = neighboringnodes(mp)
    F = get_kernel(interpolation(mp))
    xₚ = getx(pt)
    D = zero(Vec{dim, T}) # diagonal entries of M
    copyto!(mp.N, values(F, xₚ, mesh))

    @inbounds @simd for ip in eachindex(indices)
        i = indices[ip]
        w = mp.N[ip]
        xᵢ = mesh[i]
        D += w * (xᵢ - xₚ) .* (xᵢ - xₚ)
        mp.∇N[ip] = w * (xᵢ - xₚ)
    end

    D⁻¹ = inv.(D)
    broadcast!(.*, mp.∇N, mp.∇N, D⁻¹)
end

function fast_update_property_nearbounds!(mp::MPValues{<: WLS{<: BSpline}}, pt, mesh::CartesianMesh{dim, T}, filter::AbstractArray{Bool}) where {dim, T}
    indices = neighboringnodes(mp)
    it = interpolation(mp)
    F = get_kernel(it)
    M = zero(Mat{dim+1, dim+1, T})
    xₚ = getx(pt)

    @inbounds @simd for ip in eachindex(indices)
        i = indices[ip]
        xᵢ = mesh[i]
        w = value(F, xₚ, mesh, i) * filter[i]
        p = [1; xᵢ-xₚ]
        M += w * p ⊗ p
        mp.N[ip] = w
    end

    M⁻¹ = inv(M)

    @inbounds @simd for ip in eachindex(indices)
        i = indices[ip]
        xᵢ = mesh[i]
        wq = mp.N[ip] * (M⁻¹ ⋅ [1;xᵢ-xₚ])
        mp.N[ip] = wq[1]
        mp.∇N[ip] = @Tensor wq[2:end]
    end
end

Base.show(io::IO, wls::WLS) = print(io, WLS, "(", get_kernel(wls), ")")
