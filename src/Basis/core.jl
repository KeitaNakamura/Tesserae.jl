# Basis interfaces

"""
    Basis

Abstract type for basis functions used to compute [`BasisWeight`](@ref)s.
"""
abstract type Basis end

"""
    Kernel <: Basis

Abstract subtype for compact-support kernels.
"""
abstract type Kernel <: Basis end

"""
    nodal_basis_jet(order, basis, pt, mesh, i)

Return the value and spatial derivatives up to `order` of the `i`th basis
function evaluated at `pt`.
"""
function nodal_basis_jet end

"""
    jet(order, f, x, args...)

Return the value and derivatives up to `order` in `f`'s natural local or
reference coordinate system.
"""
@inline function jet(::Order{k}, f, x, args...) where {k}
    reverse(∂{k}(y -> value(f, y, args...), x, :all))
end

"""
    axis_jet_args(kernel, pt, mesh, i)

Extra arguments for the 1-D `jet` evaluation, as one tuple per Cartesian axis.
Kernels whose 1-D value depends on the local coordinate alone need none.
"""
@inline axis_jet_args(::Kernel, pt, mesh::CartesianMesh{dim}, i) where {dim} = nfill((), Val(dim))

# `@generated` is required because `Order(a-1)` must see `a` as a literal to
# become a type parameter, which `ntuple(f, Val(N))` cannot supply.
@generated function separable_nodal_basis_jet(order::Order{k}, kernel, pt, mesh::CartesianMesh{dim}, i) where {dim, k}
    quote
        @_inline_meta
        x = getx(pt)
        h⁻¹ = spacing_inv(mesh)
        ξ = (x - mesh[i]) * h⁻¹
        args = axis_jet_args(kernel, pt, mesh, i) # must precede the `@ntuple` whose loop variable shadows `i`
        vals1d = @ntuple $dim d -> jet(order, kernel, ξ[d], args[d]...)
        vals = @ntuple $(k+1) a -> only(prod_each_dimension(Order(a-1), vals1d...))
        @ntuple $(k+1) i -> vals[i]*h⁻¹^(i-1)
    end
end

#=
Standard Cartesian basis extension:
* Tesserae.support_width(basis)
* Tesserae.supportnodes(basis, pt, mesh)
* Tesserae.update_basis_values!(bw::BasisWeight, basis, pt, mesh)

A basis whose support is not a fixed Cartesian block overrides
`allocate_basis_values`, or defines `update_basis_weight!` for its own
`BasisWeight` as `CPDI` does, updating `supportnodes(bw)` and
`nodal_basis_values(bw, order)` with matching local indices. It then forfeits
every feature needing a fixed support width: `Partition` block sizing in
`@P2G`, and the sparsity radius used by
`create_sparse_matrix`/`create_block_sparse_matrix`.
=#

# Reported here rather than as a bare `MethodError`, `support_width` also being
# read outside `Basis/` by the block-size check and the sparsity radius.
function support_width(basis::Basis)
    error("$(nameof(typeof(basis))) does not define `Tesserae.support_width`. It sizes the default `BasisWeight` storage, and is also needed by `Partition` in `@P2G` and by `create_sparse_matrix`/`create_block_sparse_matrix`. Define `Tesserae.support_width` for it, or, if its support is not a fixed Cartesian block, override `Tesserae.allocate_basis_values` as `CPDI` does.")
end

initial_supportnodes(::Basis, ::CartesianMesh{dim}) where {dim} = EmptyCartesianIndices(Val(dim))

@inline derivative_count(::Order{k}) where {k} = k + 1
