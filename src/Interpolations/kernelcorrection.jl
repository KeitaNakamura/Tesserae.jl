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

@inline function update_property!(iw::InterpolationWeight, kc::KernelCorrection, pt, mesh::CartesianMesh, filter::AbstractArray{Bool} = Trues(size(mesh)))
    indices = neighboringnodes(iw)
    isnearbounds = size(values(iw,1)) != size(indices) || !alltrue(filter, indices)
    if isnearbounds
        update_property!(iw, WLS(get_kernel(kc), get_polynomial(kc)), pt, mesh, filter)
    else
        update_property!(iw, get_kernel(kc), pt, mesh)
    end
end

Base.show(io::IO, kc::KernelCorrection) = print(io, KernelCorrection, "(", get_kernel(kc), ", ", get_polynomial(kc), ")")
