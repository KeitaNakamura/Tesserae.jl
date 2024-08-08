"""
    KernelCorrection(::Kernel)

Kernel correction method [^KC] modifies kernels to achieve stable simulations near boundaries.
Available kernels include `BSpline`s and `uGIMP`.

[^KC]: [Nakamura, K., Matsumura, S., & Mizutani, T. (2023). Taylor particle-in-cell transfer and kernel correction for material point method. *Computer Methods in Applied Mechanics and Engineering*, 403, 115720.](https://doi.org/10.1016/j.cma.2022.115720)
"""
struct KernelCorrection{K <: Kernel, P <: Polynomial} <: Interpolation
    kernel::K
    poly::P
end

KernelCorrection(k::Kernel) = KernelCorrection(k, Polynomial(Linear()))

get_kernel(kc::KernelCorrection) = kc.kernel
get_polynomial(kc::KernelCorrection) = kc.poly
gridspan(kc::KernelCorrection) = gridspan(get_kernel(kc))
@inline neighboringnodes(kc::KernelCorrection, pt, mesh::CartesianMesh) = neighboringnodes(get_kernel(kc), pt, mesh)

# general version
@inline function update_property!(mp::MPValue, it::KernelCorrection, pt, mesh::CartesianMesh, filter::AbstractArray{Bool} = Trues(size(mesh)))
    indices = neighboringnodes(mp)
    isnearbounds = size(mp.w) != size(indices) || !alltrue(filter, indices)
    if isnearbounds
        update_property!(mp, WLS(get_kernel(it), get_polynomial(it)), pt, mesh, filter)
    else
        @inbounds @simd for ip in eachindex(indices)
            i = indices[ip]
            set_kernel_values!(mp, ip, value(derivative_order(mp), get_kernel(it), pt, mesh, i))
        end
    end
end

# fast version for B-spline kernels
@inline function update_property!(mp::MPValue, it::KernelCorrection{<: AbstractBSpline}, pt, mesh::CartesianMesh, filter::AbstractArray{Bool} = Trues(size(mesh)))
    indices = neighboringnodes(mp)
    isnearbounds = size(mp.w) != size(indices) || !alltrue(filter, indices)
    if isnearbounds
        update_property_general!(mp, WLS(get_kernel(it), get_polynomial(it)), pt, mesh, filter)
    else
        set_kernel_values!(mp, values(derivative_order(mp), get_kernel(it), getx(pt), mesh))
    end
end

Base.show(io::IO, kc::KernelCorrection) = print(io, KernelCorrection, "(", get_kernel(kc), ", ", get_polynomial(kc), ")")
