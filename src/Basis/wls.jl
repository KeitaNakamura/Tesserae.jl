"""
    WLS(kernel)

`WLS` performs a local weighted least squares fit for the `kernel`.
This results in the same kernel used in moving least squares MPM[^MLSMPM].
`kernel` is one of [`BSpline`](@ref) and [`uGIMP`](@ref).

[^MLSMPM]: [Hu, Y., Fang, Y., Ge, Z., Qu, Z., Zhu, Y., Pradhana, A. and Jiang, C., 2018. A moving least squares material point method with displacement discontinuity and two-way rigid body coupling. ACM Transactions on Graphics (TOG), 37(4), pp.1-14.](https://doi.org/10.1145/3197517.3201293)
"""
struct WLS{K <: Kernel, P <: CorrectionPolynomial} <: Basis
    kernel::K
    poly::P
end

WLS(k::Kernel) = WLS(k, Polynomial(Linear()))
WLS(::Kernel, poly::Polynomial) = throw(ArgumentError(unsupported_correction_polynomial("WLS", poly)))

support_width(wls::WLS) = support_width(wls.kernel)
@inline supportnodes(wls::WLS, pt, mesh::CartesianMesh) = supportnodes(wls.kernel, pt, mesh)
@inline supports_filtered_updates(::WLS) = true

@inline update_basis_values!(bw::BasisWeight, wls::WLS, pt, mesh::CartesianMesh) =
    update_basis_values!(bw, wls, pt, mesh, Trues(size(mesh)))
@inline function update_basis_values!(bw::BasisWeight, wls::WLS, pt, mesh::CartesianMesh, filter::AbstractArray{Bool})
    update_wls_values!(bw, wls, pt, mesh, filter)
end

# a bit faster implementation for B-splines
@inline function update_basis_values!(bw::BasisWeight, wls::WLS{<: Union{BSpline{Quadratic}, BSpline{Cubic}}, <: Polynomial{Linear}}, pt, mesh::CartesianMesh, filter::AbstractArray{Bool})
    indices = supportnodes(bw)
    if has_full_support(bw, indices, filter)
        kernel = wls.kernel
        @inbounds for ip in eachindex(indices)
            nodal_basis_values(bw, Order(0))[ip] = only(nodal_basis_jet(Order(0), kernel, pt, mesh, indices[ip]))
        end
        apply_wls_correction!(bw, wls, pt, mesh, full_support_moment_matrix_inv(kernel, mesh))
    else
        update_wls_values!(bw, wls, pt, mesh, filter)
    end
end
# Over its full support a degree-`n` cardinal B-spline has unit mass, zero first
# moment, and second moment `(n+1)h²/12`, independent of where the particle sits
# in the cell, so the linear-basis moment matrix is diagonal and known in closed
# form. That holds from `n = 2` only: for `n = 1` the second moment varies with
# the position as ξ(1-ξ), so `Linear` is excluded rather than given a wrong
# matrix. `12//(n+1)` is evaluated as a rational before dividing by `h²`, so
# the quotient is the exact 4 or 3 the two hand-written entries used and the
# result stays in the mesh scalar type — a `12/(n+1)` float quotient would
# promote the matrix to Float64, which Metal kernels cannot compile.
@inline function full_support_moment_matrix_inv(::AbstractBSpline{Degree{n}}, mesh::CartesianMesh{dim}) where {n, dim}
    n ≥ 2 || throw(ArgumentError("the full-support moment matrix is position-dependent below degree 2"))
    diagm([1; ones(Vec{dim,Int}) * (12//(n+1)) / spacing(mesh)^2])
end

@inline function update_wls_values!(bw::BasisWeight, wls::WLS, pt, mesh::CartesianMesh, filter::AbstractArray{Bool})
    indices = supportnodes(bw)
    kernel = wls.kernel
    poly = wls.poly
    xₚ = getx(pt)

    M = fastsum(eachindex(indices)) do ip
        @inbounds begin
            i = indices[ip]
            xᵢ = mesh[i]
            w = nodal_basis_values(bw, Order(0))[ip] = only(nodal_basis_jet(Order(0), kernel, pt, mesh, i)) * filter[i]
            P = value(poly, xᵢ - xₚ)
            w * P ⊗ P
        end
    end

    apply_wls_correction!(bw, wls, pt, mesh, inv(M))
end

@inline function apply_wls_correction!(bw::BasisWeight, wls::WLS, pt, mesh::CartesianMesh, M⁻¹)
    indices = supportnodes(bw)
    poly = wls.poly
    xₚ = getx(pt)

    P₀__ = jet(derivative_order(bw), poly, zero(xₚ))
    @inbounds for ip in eachindex(indices)
        i = indices[ip]
        xᵢ = mesh[i]
        w = nodal_basis_values(bw, Order(0))[ip]
        P = value(poly, xᵢ - xₚ)
        wq = w * (M⁻¹ * P)
        set_values!(bw, ip, map(P₀->wq⊡P₀, P₀__))
    end
end

function update_basis_values!(bw::BasisWeight, wls::WLS{<: Union{BSpline{Quadratic}, BSpline{Cubic}, BSpline{Quartic}, BSpline{Quintic}}, Polynomial{MultiLinear}}, pt, mesh::CartesianMesh{dim}, filter::AbstractArray{Bool}) where {dim}
    # Masked cases require the general moment matrix.
    filter isa Trues || return update_wls_values!(bw, wls, pt, mesh, filter)

    # For MultiLinear, decompose into axis-wise Linear bases.
    wls_1d = WLS(wls.kernel, Polynomial(Linear()))
    if dim == 1
        return update_basis_values!(bw, wls_1d, pt, mesh, filter)
    end

    T = scalartype(bw)
    order = derivative_order(bw)
    vals_axes = ntuple(Val(dim)) do d
        mesh_1d = axismesh(mesh, d)
        vals_1d = allocate_static_basis_values(@NamedTuple{w::T}, wls_1d, Val(1); derivative=order)
        indices_1d = CartesianIndices((supportnodes(bw).indices[d],))
        bw_1d = BasisWeight(wls_1d, vals_1d, Scalar(indices_1d), order)
        # Must be inlined: creates/updates a small StaticArray (MVector/MArray) on the GPU.
        # If not inlined, the temporary may escape and trigger dynamic allocation (gpu_gc_pool_alloc).
        update_basis_values!(bw_1d, wls_1d, Vec(getx(pt)[d]), mesh_1d, Trues(size(mesh_1d)))
        # Get scalar value from Vec{1} for each property.
        scalarize_axis_values(order, bw_1d)
    end
    set_values!(bw, tensor_product_axis_values(order, vals_axes))
end
@inline function scalarize_axis_values(::Order{k}, bw) where {k}
    ntuple(a -> map(only, Tuple(nodal_basis_values(bw, Order(a-1)))), Val(k+1))
end
@generated function tensor_product_axis_values(::Order{k}, vals) where {k}
    quote
        @_inline_meta
        @ntuple $(k+1) a -> prod_each_dimension(Order(a-1), vals...)
    end
end

Base.show(io::IO, wls::WLS) = print(io, WLS, "(", wls.kernel, ", ", wls.poly, ")")

# Deferred evaluation. The moment matrix is a per-particle quantity, so it is
# built once in the support loop's preamble and every node then reads it; that is
# the whole of what made these bases look undeferrable. `full` records whether
# the correction applies at all, which is what `KernelCorrection` branches on --
# it is carried rather than acted on here so both arms have the same type.
@inline function wls_deferred_state(order::Order, kernel, poly, pt, mesh::CartesianMesh, window, filter, full::Bool)
    xₚ = getx(pt)
    P₀__ = jet(order, poly, zero(xₚ))
    P₀ = value(poly, zero(xₚ))
    full && return (true, zero(P₀ ⊗ P₀), P₀__)
    M = fastsum(eachindex(window)) do ip
        @inbounds begin
            i = window[ip]
            w = only(nodal_basis_jet(Order(0), kernel, pt, mesh, i)) * filterpasses(filter, i)
            P = value(poly, mesh[i] - xₚ)
            w * P ⊗ P
        end
    end
    (false, inv(M), P₀__)
end

@inline function wls_deferred_jet(kernel, poly, state, pt, mesh::CartesianMesh, window, filter, ip)
    @_propagate_inbounds_meta
    _, M⁻¹, P₀__ = state
    i = window[ip]
    w = only(nodal_basis_jet(Order(0), kernel, pt, mesh, i)) * filterpasses(filter, i)
    P = value(poly, mesh[i] - getx(pt))
    wq = w * (M⁻¹ * P)
    map(P₀ -> wq ⊡ P₀, P₀__)
end

@inline deferred_particle_state(order::Order, wls::WLS, pt, mesh::CartesianMesh, window, filter) =
    wls_deferred_state(order, wls.kernel, wls.poly, pt, mesh, window, filter, false)
@inline deferred_node_jet(::Order, wls::WLS, state::Tuple, pt, mesh::CartesianMesh, window, filter, ip) =
    (@_propagate_inbounds_meta; wls_deferred_jet(wls.kernel, wls.poly, state, pt, mesh, window, filter, ip))

can_defer_basis(::Type{<: WLS}) = true
check_deferred_basis(::WLS) = nothing
needs_filter(::WLS) = true
