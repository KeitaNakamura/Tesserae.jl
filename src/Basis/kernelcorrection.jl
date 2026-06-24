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
struct KernelCorrection{K <: Kernel, P <: Polynomial} <: Basis
    kernel::K
    poly::P
end

KernelCorrection(k::Kernel) = KernelCorrection(k, Polynomial(MultiLinear()))

get_kernel(kc::KernelCorrection) = kc.kernel
get_polynomial(kc::KernelCorrection) = kc.poly
kernel_support(kc::KernelCorrection) = kernel_support(get_kernel(kc))
@inline supportnodes(kc::KernelCorrection, pt, mesh::CartesianMesh) = supportnodes(get_kernel(kc), pt, mesh)

@inline function update_basis_values!(bw::BasisWeight, kc::KernelCorrection, pt, mesh::CartesianMesh, filter::AbstractArray{Bool} = Trues(size(mesh)))
    indices = supportnodes(bw)
    if has_full_support(bw, indices, filter)
        update_basis_values_full!(bw, kc, pt, mesh)
    else
        update_basis_values_truncated!(bw, kc, pt, mesh, filter)
    end
end

function update_basis_values_truncated!(bw::BasisWeight, kc::KernelCorrection, pt, mesh, filter)
    update_basis_values!(bw, WLS(get_kernel(kc), get_polynomial(kc)), pt, mesh, filter)
end
@inline function update_basis_values_full!(bw::BasisWeight, kc::KernelCorrection, pt, mesh)
    update_basis_values!(bw, get_kernel(kc), pt, mesh)
end

Base.show(io::IO, kc::KernelCorrection) = print(io, KernelCorrection, "(", get_kernel(kc), ", ", get_polynomial(kc), ")")
