"""
    WLS(kernel)

`WLS` performs a local weighted least squares fit for the `kernel`.
This results in the same kernel used in moving least squares MPM[^MLSMPM].
`kernel` is one of [`BSpline`](@ref) and [`uGIMP`](@ref).

[^MLSMPM]: [Hu, Y., Fang, Y., Ge, Z., Qu, Z., Zhu, Y., Pradhana, A. and Jiang, C., 2018. A moving least squares material point method with displacement discontinuity and two-way rigid body coupling. ACM Transactions on Graphics (TOG), 37(4), pp.1-14.](https://doi.org/10.1145/3197517.3201293)
"""
struct WLS{K <: Kernel, P <: Polynomial} <: Basis
    kernel::K
    poly::P
end

WLS(k::Kernel) = WLS(k, Polynomial(Linear()))

get_kernel(wls::WLS) = wls.kernel
get_polynomial(wls::WLS) = wls.poly
kernel_support(wls::WLS) = kernel_support(get_kernel(wls))
@inline supportnodes(wls::WLS, pt, mesh::CartesianMesh) = supportnodes(get_kernel(wls), pt, mesh)

@inline function update_property!(bw::BasisWeight, wls::WLS, pt, mesh::CartesianMesh, filter::AbstractArray{Bool} = Trues(size(mesh)))
    update_property_general!(bw, wls, pt, mesh, filter)
end

# a bit faster implementation for B-splines
@inline function update_property!(bw::BasisWeight, wls::WLS{<: Union{BSpline{Quadratic}, BSpline{Cubic}}, <: Polynomial{Linear}}, pt, mesh::CartesianMesh, filter::AbstractArray{Bool} = Trues(size(mesh)))
    indices = supportnodes(bw)
    if has_full_support(bw, indices, filter)
        kernel = get_kernel(wls)
        @inbounds for ip in eachindex(indices)
            basis_values(bw, Order(0))[ip] = only(basis_jet(Order(0), kernel, pt, mesh, indices[ip]))
        end
        update_property_after_moment_matrix!(bw, wls, pt, mesh, moment_matrix_inv(kernel, mesh))
    else
        update_property_general!(bw, wls, pt, mesh, filter)
    end
end
@inline moment_matrix_inv(::BSpline{Quadratic}, mesh::CartesianMesh{dim}) where {dim} = diagm([1; ones(Vec{dim,Int}) * 4/spacing(mesh)^2])
@inline moment_matrix_inv(::BSpline{Cubic}, mesh::CartesianMesh{dim}) where {dim} = diagm([1; ones(Vec{dim,Int}) * 3/spacing(mesh)^2])

@inline function update_property_general!(bw::BasisWeight, wls::WLS, pt, mesh::CartesianMesh, filter::AbstractArray{Bool})
    indices = supportnodes(bw)
    kernel = get_kernel(wls)
    poly = get_polynomial(wls)
    xₚ = getx(pt)

    M = fastsum(eachindex(indices)) do ip
        @inbounds begin
            i = indices[ip]
            xᵢ = mesh[i]
            w = basis_values(bw, Order(0))[ip] = only(basis_jet(Order(0), kernel, pt, mesh, i)) * filter[i]
            P = value(poly, xᵢ - xₚ)
            w * P ⊗ P
        end
    end

    update_property_after_moment_matrix!(bw, wls, pt, mesh, inv(M))
end

@inline function update_property_after_moment_matrix!(bw::BasisWeight, wls::WLS, pt, mesh::CartesianMesh, M⁻¹)
    indices = supportnodes(bw)
    poly = get_polynomial(wls)
    xₚ = getx(pt)

    P₀__ = jet(derivative_order(bw), poly, zero(xₚ))
    @inbounds for ip in eachindex(indices)
        i = indices[ip]
        xᵢ = mesh[i]
        w = basis_values(bw, Order(0))[ip]
        P = value(poly, xᵢ - xₚ)
        wq = w * (M⁻¹ * P)
        set_values!(bw, ip, map(P₀->wq⊡P₀, P₀__))
    end
end

function update_property!(bw::BasisWeight, wls::WLS{<: Union{BSpline{Quadratic}, BSpline{Cubic}, BSpline{Quartic}, BSpline{Quintic}}, Polynomial{MultiLinear}}, pt, mesh::CartesianMesh{dim}, filter::AbstractArray{Bool} = Trues(size(mesh))) where {dim}
    if filter isa Trues
        # For MultiLinear, we can decompose into axis-wise Linear bases.
        # If the problem is 1D, MultiLinear == Linear, so use the direct fast path.
        wls_1d = WLS(get_kernel(wls), Polynomial(Linear()))
        if dim == 1
            update_property!(bw, wls_1d, pt, mesh, filter)
        else
            # For dim > 1: decompose into 1D Linear along each axis,
            # compute axis-wise contribution, then combine by tensor product.
            T = scalartype(bw)
            order = derivative_order(bw)
            vals_axes = ntuple(Val(dim)) do d
                mesh_1d = axismesh(mesh, d)
                prop_1d = create_property(MArray, Vec{1,T}, wls_1d; derivative=order)
                indices_1d = CartesianIndices((supportnodes(bw).indices[d],))
                bw_1d = BasisWeight(wls_1d, prop_1d, Scalar(indices_1d))
                # Must be inlined: creates/updates a small StaticArray (MVector/MArray) on the GPU.
                # If not inlined, the temporary may escape and trigger dynamic allocation (gpu_gc_pool_alloc).
                update_property_general!(bw_1d, wls_1d, Vec(getx(pt)[d]), mesh_1d, Trues(size(mesh_1d)))
                # Get scalar value from Vec{1} for each property.
                _extract_scalar_values(order, bw_1d)
            end
            # Combine axis-wise results into MultiLinear tensor product.
            set_values!(bw, _prod_each_dimension(order, vals_axes))
        end
    else
        # Fallback for masked cases: use general method.
        update_property_general!(bw, wls, pt, mesh, filter)
    end
end
@inline function _extract_scalar_values(::Order{k}, bw) where {k}
    ntuple(a -> map(only, Tuple(basis_values(bw, Order(a-1)))), Val(k+1))
end
@generated function _prod_each_dimension(::Order{k}, vals) where {k}
    quote
        @_inline_meta
        @ntuple $(k+1) a -> prod_each_dimension(Order(a-1), vals...)
    end
end

Base.show(io::IO, wls::WLS) = print(io, WLS, "(", get_kernel(wls), ", ", get_polynomial(wls), ")")
