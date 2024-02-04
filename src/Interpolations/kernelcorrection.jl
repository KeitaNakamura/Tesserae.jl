"""
    KernelCorrection(::Kernel)

Kernel correction method [^KC].

This modifies kernels to achive stable simulations near boundaries.
Using kernel correction is preferred for [`QuadraticBSpline`](@ref) and [`CubicBSpline`](@ref),
and almost necessary for [`AffineTransfer`](@ref) and [`TaylorTransfer`](@ref).

[^KC]: [Nakamura, K., Matsumura, S., & Mizutani, T. (2023). Taylor particle-in-cell transfer and kernel correction for material point method. *Computer Methods in Applied Mechanics and Engineering*, 403, 115720.](https://doi.org/10.1016/j.cma.2022.115720)
"""
struct KernelCorrection{K <: Kernel} <: Interpolation
    kernel::K
end

get_kernel(kc::KernelCorrection) = kc.kernel
gridspan(kc::KernelCorrection) = gridspan(get_kernel(kc))
@inline neighbornodes(kc::KernelCorrection, pt, lattice::Lattice) = neighbornodes(get_kernel(kc), pt, lattice)

# general version
@inline function update_property!(mp::MPValues{<: KernelCorrection}, pt, lattice::Lattice, filter::AbstractArray{Bool} = Trues(size(lattice)))
    indices = neighbornodes(mp)
    isnearbounds = size(mp.N) != size(indices) || !alltrue(filter, indices)
    if isnearbounds
        update_property_nearbounds!(mp, pt, lattice, filter)
    else
        @inbounds @simd for ip in eachindex(indices)
            i = indices[ip]
            mp.N[ip], mp.∇N[ip] = value_gradient(get_kernel(interpolation(mp)), pt, lattice, i)
        end
    end
end

# fast version for B-spline kernels
@inline function update_property!(mp::MPValues{<: KernelCorrection{<: BSpline}}, pt, lattice::Lattice, filter::AbstractArray{Bool} = Trues(size(lattice)))
    indices = neighbornodes(mp)
    isnearbounds = size(mp.N) != size(indices) || !alltrue(filter, indices)
    if isnearbounds
        update_property_nearbounds!(mp, pt, lattice, filter)
    else
        values_gradients!(mp.N, mp.∇N, get_kernel(interpolation(mp)), pt, lattice)
    end
end

@inline function update_property_nearbounds!(mp::MPValues{<: KernelCorrection}, pt, lattice::Lattice, filter::AbstractArray{Bool})
    indices = neighbornodes(mp)
    F = get_kernel(interpolation(mp))
    xₚ = getx(pt)
    VecType = promote_type(eltype(lattice), typeof(xₚ))
    dim, T = length(VecType), eltype(VecType)
    M = zero(Mat{dim+1, dim+1, T})
    @inbounds for ip in eachindex(indices)
        i = indices[ip]
        xᵢ = lattice[i]
        w = value(F, pt, lattice, i) * filter[i]
        P = [1; xᵢ - xₚ]
        M += w * P ⊗ P
        mp.N[ip] = w
    end
    M⁻¹ = inv(M)
    C₁ = M⁻¹[1,1]
    C₂ = @Tensor M⁻¹[2:end,1]
    C₃ = @Tensor M⁻¹[2:end,2:end]
    @inbounds for ip in eachindex(indices)
        i = indices[ip]
        xᵢ = lattice[i]
        w = mp.N[ip]
        mp.N[ip] = (C₁ + C₂ ⋅ (xᵢ - xₚ)) * w
        mp.∇N[ip] = (C₂ + C₃ ⋅ (xᵢ - xₚ)) * w
    end
end

Base.show(io::IO, kc::KernelCorrection) = print(io, KernelCorrection, "(", get_kernel(kc), ")")
