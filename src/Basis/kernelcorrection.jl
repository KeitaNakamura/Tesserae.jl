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

# Deferred evaluation. The branch `update_basis_values!` takes per particle --
# plain kernel where the support is whole, the weighted fit where the mesh cuts
# it -- is decided once here and recorded in the state, so both arms of the node
# evaluation produce the same type.
@inline deferred_particle_state(order::Order, kc::KernelCorrection, pt, mesh::CartesianMesh, window, filter) =
    wls_deferred_state(order, kc.kernel, kc.poly, pt, mesh, window, filter,
                       all(size(window) .== support_width(kc.kernel)) && allpass(filter, window))

@inline function deferred_node_jet(order::Order, kc::KernelCorrection, state::Tuple, pt, mesh::CartesianMesh, window, filter, ip)
    @_propagate_inbounds_meta
    first(state) ? nodal_basis_jet(order, kc.kernel, pt, mesh, window[ip]) :
                   wls_deferred_jet(kc.kernel, kc.poly, state, pt, mesh, window, filter, ip)
end

can_defer_basis(::Type{<: KernelCorrection}) = true
check_deferred_basis(::KernelCorrection) = nothing
needs_filter(::KernelCorrection) = true

const SeparableKernelCorrection = KernelCorrection{<: Union{BSpline{Quadratic}, BSpline{Cubic}, BSpline{Quartic}, BSpline{Quintic}}, Polynomial{MultiLinear}}

@inline function deferred_particle_state(order::Order, kc::SeparableKernelCorrection, pt, mesh::CartesianMesh, window, ::Nothing)
    (all(size(window) .== support_width(kc.kernel)), wls_axis_jets(order, kc.kernel, pt, mesh, window))
end

@inline function deferred_node_jet(order::Order, kc::SeparableKernelCorrection, state::Tuple, pt, mesh::CartesianMesh, window, ::Nothing, ip)
    @_propagate_inbounds_meta
    full, axisjets = state
    full ? nodal_basis_jet(order, kc.kernel, pt, mesh, window[ip]) :
           wls_axis_jet_at(order, axisjets, node_offsets(window, ip))
end
