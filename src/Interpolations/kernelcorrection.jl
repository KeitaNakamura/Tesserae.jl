"""
    KernelCorrection(kernel)

`KernelCorrection`[^KC] modifies `kernel` to achieve stable simulations near boundaries.
The corrected kernel satisfies not only the partition of unity, ``\\sum_i w_{ip} = 1``,
but also the linear field reproduction, ``\\sum_i w_{ip} \\bm{x}_i = \\bm{x}_p``, near boundaries.
In the implementation, this simply applies [`WLS`](@ref) near boundaries.
`kernel` is one of [`BSpline`](@ref) and [`uGIMP`](@ref).
See also [`SteffenBSpline`](@ref).

[^KC]: [Nakamura, K., Matsumura, S., & Mizutani, T. (2023). Taylor particle-in-cell transfer and kernel correction for material point method. *Computer Methods in Applied Mechanics and Engineering*, 403, 115720.](https://doi.org/10.1016/j.cma.2022.115720)
"""
struct KernelCorrection{K <: Kernel, P <: Polynomial} <: Interpolation
    kernel::K
    poly::P
end

KernelCorrection(k::Kernel) = KernelCorrection(k, Polynomial(MultiLinear()))

get_kernel(kc::KernelCorrection) = kc.kernel
get_polynomial(kc::KernelCorrection) = kc.poly
kernel_support(kc::KernelCorrection) = kernel_support(get_kernel(kc))
@inline neighboringnodes(kc::KernelCorrection, pt, mesh::CartesianMesh) = neighboringnodes(get_kernel(kc), pt, mesh)

# general version
@inline function update_property!(mp::MPValue, kc::KernelCorrection, pt, mesh::CartesianMesh, filter::AbstractArray{Bool} = Trues(size(mesh)))
    indices = neighboringnodes(mp)
    isnearbounds = size(values(mp,1)) != size(indices) || !alltrue(filter, indices)
    if isnearbounds
        update_property!(mp, WLS(get_kernel(kc), get_polynomial(kc)), pt, mesh, filter)
    else
        @inbounds @simd for ip in eachindex(indices)
            i = indices[ip]
            set_values!(mp, ip, values(derivative_order(mp), get_kernel(kc), pt, mesh, i))
        end
    end
end

# fast version for B-spline kernels
@inline function update_property!(mp::MPValue, kc::KernelCorrection{<: AbstractBSpline}, pt, mesh::CartesianMesh, filter::AbstractArray{Bool} = Trues(size(mesh)))
    indices = neighboringnodes(mp)
    isnearbounds = size(values(mp,1)) != size(indices) || !alltrue(filter, indices)
    if isnearbounds
        update_property_general!(mp, WLS(get_kernel(kc), get_polynomial(kc)), pt, mesh, filter)
    else
        set_values!(mp, values(derivative_order(mp), get_kernel(kc), getx(pt), mesh))
    end
end

Base.show(io::IO, kc::KernelCorrection) = print(io, KernelCorrection, "(", get_kernel(kc), ", ", get_polynomial(kc), ")")
