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
struct KernelCorrection{K <: Kernel, P <: CorrectionPolynomial} <: Basis
    kernel::K
    poly::P
end

KernelCorrection(k::Kernel) = KernelCorrection(k, Polynomial(MultiLinear()))
KernelCorrection(::Kernel, poly::Polynomial) = throw(ArgumentError(unsupported_correction_polynomial("KernelCorrection", poly)))

support_width(kc::KernelCorrection) = support_width(kc.kernel)
@inline supportnodes(kc::KernelCorrection, pt, mesh::CartesianMesh) = supportnodes(kc.kernel, pt, mesh)
@inline supports_filtered_updates(::KernelCorrection) = true

@inline update_basis_values!(bw::BasisWeight, kc::KernelCorrection, pt, mesh::CartesianMesh) =
    update_basis_values!(bw, kc, pt, mesh, Trues(size(mesh)))
@inline function update_basis_values!(bw::BasisWeight, kc::KernelCorrection, pt, mesh::CartesianMesh, filter::AbstractArray{Bool})
    indices = supportnodes(bw)
    if has_full_support(bw, indices, filter)
        update_basis_values!(bw, kc.kernel, pt, mesh)
    else
        update_basis_values!(bw, WLS(kc.kernel, kc.poly), pt, mesh, filter)
    end
end

Base.show(io::IO, kc::KernelCorrection) = print(io, KernelCorrection, "(", kc.kernel, ", ", kc.poly, ")")
