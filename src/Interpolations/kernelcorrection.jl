"""
    KernelCorrection(::Kernel)

Kernel correction method [^KC] modifies kernels to achieve stable simulations near boundaries.
Available kernels include `BSpline`s and `uGIMP`.

[^KC]: [Nakamura, K., Matsumura, S., & Mizutani, T. (2023). Taylor particle-in-cell transfer and kernel correction for material point method. *Computer Methods in Applied Mechanics and Engineering*, 403, 115720.](https://doi.org/10.1016/j.cma.2022.115720)
"""
struct KernelCorrection{K <: Kernel, P <: AbstractPolynomial} <: Interpolation
    kernel::K
    poly::P
end

KernelCorrection(k::Kernel) = KernelCorrection(k, LinearPolynomial())

get_kernel(kc::KernelCorrection) = kc.kernel
get_polynomial(kc::KernelCorrection) = kc.poly
gridspan(kc::KernelCorrection) = gridspan(get_kernel(kc))
@inline surroundingnodes(kc::KernelCorrection, pt, mesh::CartesianMesh) = surroundingnodes(get_kernel(kc), pt, mesh)

# general version
@inline function update_property!(mp::MPValues{<: KernelCorrection}, pt, mesh::CartesianMesh, filter::AbstractArray{Bool} = Trues(size(mesh)))
    indices = surroundingnodes(mp)
    isnearbounds = size(mp.N) != size(indices) || !alltrue(filter, indices)
    if isnearbounds
        update_property_nearbounds!(mp, pt, mesh, filter)
    else
        @inbounds @simd for ip in eachindex(indices)
            i = indices[ip]
            mp.N[ip], mp.∇N[ip] = value_gradient(get_kernel(interpolation(mp)), pt, mesh, i)
        end
    end
end

# fast version for B-spline kernels
@inline function update_property!(mp::MPValues{<: KernelCorrection{<: BSpline}}, pt, mesh::CartesianMesh, filter::AbstractArray{Bool} = Trues(size(mesh)))
    indices = surroundingnodes(mp)
    isnearbounds = size(mp.N) != size(indices) || !alltrue(filter, indices)
    if isnearbounds
        update_property_nearbounds!(mp, pt, mesh, filter)
    else
        map(copyto!, (mp.N, mp.∇N), values(get_kernel(interpolation(mp)), pt, mesh, :withgradient))
    end
end

@inline function update_property_nearbounds!(mp::MPValues{<: KernelCorrection}, pt, mesh::CartesianMesh{dim}, filter::AbstractArray{Bool}) where {dim}
    indices = surroundingnodes(mp)
    kernel = get_kernel(interpolation(mp))
    poly = get_polynomial(interpolation(mp))
    xₚ = getx(pt)
    VecType = promote_type(eltype(mesh), typeof(xₚ))
    L, T = value_length(poly, xₚ), eltype(VecType)
    M = zero(Mat{L, L, T})
    @inbounds for ip in eachindex(indices)
        i = indices[ip]
        xᵢ = mesh[i]
        w = value(kernel, pt, mesh, i) * filter[i]
        P = value(poly, xᵢ - xₚ)
        M += w * P ⊗ P
        mp.N[ip] = w
    end
    M⁻¹ = inv(M)
    @inbounds for ip in eachindex(indices)
        i = indices[ip]
        xᵢ = mesh[i]
        w = mp.N[ip]
        P = value(poly, xᵢ - xₚ)
        wq = w * (M⁻¹ ⋅ P)
        # P₀, ∇P₀ = value(poly, zero(xₚ), :withgradient)
        mp.N[ip] = wq[1] # wq ⋅ P₀
        mp.∇N[ip] = @Tensor wq[2:1+dim] # wq ⋅ ∇P₀
    end
end

Base.show(io::IO, kc::KernelCorrection) = print(io, KernelCorrection, "(", get_kernel(kc), ", ", get_polynomial(kc), ")")
