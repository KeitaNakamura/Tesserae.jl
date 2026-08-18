# -----------------------------------------------------------------------------
#  Basis weights
# -----------------------------------------------------------------------------

# ---- basis ----

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
initial_supportnodes(shape::Shape, mesh::FEMesh) = zero(SVector{nlocalnodes(shape), Int})

function basis_property_type(::Type{T}, name) where {T}
    NamedTuple{(basis_value_name(Order(0), name),), Tuple{T}}
end

function allocate_basis_values(::Type{Prop}, basis, ::Val{dim}; derivative::Order) where {Prop <: NamedTuple, dim}
    map(Array, allocate_static_basis_values(Prop, basis, Val(dim); derivative))
end
function allocate_static_basis_values(::Type{Prop}, basis::Basis, ::Val{dim}; kwargs...) where {Prop <: NamedTuple, dim}
    A = MArray{Tuple{nfill(support_width(basis), Val(dim))...}}
    _allocate_basis_values(A, Prop, Val(dim); kwargs...)
end
function allocate_static_basis_values(::Type{Prop}, shape::Shape, ::Val{dim}; kwargs...) where {Prop <: NamedTuple, dim}
    A = MArray{Tuple{nlocalnodes(shape)}}
    _allocate_basis_values(A, Prop, Val(dim); kwargs...)
end

function _allocate_basis_values(::Type{A}, ::Type{Prop}, ::Val{dim}; derivative::Order) where {A, Prop <: NamedTuple, dim}
    map(v -> fill(v, A), basis_value_zeros(Prop, Val(dim), derivative))
end

@generated function basis_value_zeros(::Type{Prop}, ::Val{dim}, ::Order{k}) where {Prop <: NamedTuple, dim, k}
    prop_names = fieldnames(Prop)
    isempty(prop_names) && return :(throw(ArgumentError("basis-weight property type must have at least one field")))
    T = fieldtype(Prop, 1)

    jet_names = ntuple(i -> basis_value_name(Order(i-1), Val(first(prop_names))), k+1)
    names = (jet_names..., Base.tail(prop_names)...)
    allunique(names) || return :(throw(ArgumentError("generated basis-value names overlap custom property names")))
    jet_zeros = [:(zero_basis_value(Vec{$dim, $T}, Order($(i-1)))) for i in 1:k+1]
    custom_zeros = [:(zero_recursive($(fieldtype(Prop, i)))) for i in 2:fieldcount(Prop)]

    quote
        @_inline_meta
        NamedTuple{$names}(($(jet_zeros...), $(custom_zeros...)))
    end
end

# For code generators only, where `dim` and `k` are literals: `zero_basis_value`
# below spells the type out instead, because routing it through a call loses inference.
jet_value_type(::Order{1}, dim) = Vec{dim}
jet_value_type(::Order{k}, dim) where {k} = Tensor{Tuple{@Symmetry{fill(dim, k)...}}}

zero_basis_value(::Type{Vec{dim, T}}, ::Order{0}) where {dim, T} = zero(T)
zero_basis_value(::Type{Vec{dim, T}}, ::Order{1}) where {dim, T} = zero(Vec{dim, T})
zero_basis_value(::Type{Vec{dim, T}}, ::Order{k}) where {dim, T, k} = zero(Tensor{Tuple{@Symmetry{ntuple(i->dim, k)...}}, T})
basis_value_name(::Order{0}, ::Val{name}) where {name} = name
for (k, nabla) in enumerate((:∇, :∇², :∇³, :∇⁴, :∇⁵, :∇⁶, :∇⁷, :∇⁸, :∇⁹))
    @eval begin
        basis_value_name(::Order{$k}, ::Val{name}) where {name} = Symbol($(QuoteNode(nabla)), name)
    end
end

@inline function prod_each_dimension(::Order{0}, vals::Vararg{Tuple, dim}) where {dim}
    tuple_otimes(ntuple(d -> vals[d][1], Val(dim)))
end
@generated function prod_each_dimension(::Order{k}, vals::Vararg{Tuple, dim}) where {k, dim}
    TT = jet_value_type(Order(k), dim)
    v = Array{Expr}(undef, size(TT))
    for I in CartesianIndices(v)
        ex = Expr(:tuple)
        for i in 1:dim
            j = count(==(i), Tuple(I)) + 1
            push!(ex.args, :(vals[$i][$j]))
        end
        v[I] = ex
    end
    quote
        @_inline_meta
        v = $(Expr(:tuple, v[Tensorial.independent_to_component_map(TT)]...))
        map($TT, map(tuple_otimes, v)...)
    end
end
@inline tuple_otimes(x::Tuple) = SArray(⊗(map(Vec, x)...))

# ---- BasisWeight ----

"""
    BasisWeight([T,] basis, mesh; derivative=Order(1), name=Val(:w))
    BasisWeight(Prop, basis, mesh; derivative=Order(1))

`BasisWeight` stores basis function values and their spatial derivatives.

In the first form, the scalar type defaults to `Float64`. `name` sets the name
of the basis value, and `derivative` sets the highest derivative order to
store.

In the second form, `Prop` must be a non-empty `NamedTuple` type. Its first
field defines the basis value name and scalar type. Fields generated through
`derivative` are inserted after the first field, followed by the remaining
fields of `Prop` in their original order.

For example, `@NamedTuple{N::Float32, ψ::Vec{2,Float64}}` with
`derivative=Order(2)` creates storage for `N`, `∇N`, `∇²N`, and `ψ`.
Custom field names must not overlap generated derivative names.

```jldoctest
julia> mesh = CartesianMesh(1.0, (0,5), (0,5));

julia> xₚ = Vec(2.2, 3.4); # particle position

julia> bw = BasisWeight(BSpline(Quadratic()), mesh);

julia> update!(bw, xₚ, mesh) # update `bw` at position `xₚ` in `mesh`
BasisWeight:
  Basis: BSpline(Quadratic())
  Basis values: w::Matrix{Float64}, ∇w::Matrix{Vec{2, Float64}}
  Support nodes: CartesianIndices((2:4, 3:5))

julia> sum(bw.w) ≈ 1 # partition of unity
true

julia> nodeindices = supportnodes(bw) # grid indices within a particles' local domain
CartesianIndices((2:4, 3:5))

julia> sum(eachindex(nodeindices)) do ip # linear field reproduction
           i = nodeindices[ip]
           bw.w[ip] * mesh[i]
       end ≈ xₚ
true
```
"""
struct BasisWeight{B, Vals <: NamedTuple, Indices <: AbstractArray{<: Any}, O <: Order}
    basis::B
    vals::Vals
    indices::Indices
    order::O
end

function _basis_weight(::Type{Prop}, basis, mesh::AbstractMesh{dim}; derivative::Order=Order(1)) where {Prop <: NamedTuple, dim}
    vals = allocate_basis_values(Prop, basis, Val(dim); derivative)
    indices = initial_supportnodes(basis, mesh)
    BasisWeight(basis, vals, fill(indices), derivative)
end
function _basis_weight(::Type{T}, basis, mesh::AbstractMesh; derivative::Order=Order(1), name=Val(:w)) where {T}
    _basis_weight(basis_property_type(T, name), basis, mesh; derivative)
end

BasisWeight(::Type{T}, basis::Basis, mesh::CartesianMesh; kwargs...) where {T} = _basis_weight(T, basis, mesh; kwargs...)
BasisWeight(basis::Basis, mesh::CartesianMesh; kwargs...) = _basis_weight(Float64, basis, mesh; kwargs...)

BasisWeight(::Type{T}, mesh::FEMesh; kwargs...) where {T} = _basis_weight(T, basis(mesh), mesh; kwargs...)
BasisWeight(mesh::FEMesh; kwargs...) = BasisWeight(Float64, mesh; kwargs...)

Base.propertynames(bw::BasisWeight) = propertynames(getfield(bw, :vals))
@inline function Base.getproperty(bw::BasisWeight, name::Symbol)
    getproperty(getfield(bw, :vals), name)
end

"""
    nodal_basis_values(weight, order)

Return the array that stores nodal basis values (`Order(0)`) or their
derivatives of the requested `order`.

This array may be larger than `supportnodes(weight)`. After an update, only
entries corresponding to `eachindex(supportnodes(weight))` are valid.
"""
@generated function nodal_basis_values(bw::BasisWeight{B, Vals, Indices, Order{n}}, ::Order{k}) where {B, Vals, Indices, n, k}
    k ≤ n || return :(throw(ArgumentError("basis weight stores derivatives through Order($n), got Order($k)")))
    :(getfield(bw, :vals)[$(k+1)])
end

@inline scalartype(bw::BasisWeight) = eltype(nodal_basis_values(bw, Order(0)))

"""
    basis(mesh::FEMesh)
    basis(mesh::IGAMesh)
    basis(weight)

Return the basis associated with a mesh or basis-weight storage.
"""
@inline basis(mesh::FEMesh) = cellshape(mesh)
@inline basis(bw::BasisWeight) = getfield(bw, :basis)

"""
    supportnodes(weight[, domain])

Return the nodes in the support of a [`BasisWeight`](@ref).
When `domain` is a `Grid`, `SpGrid`, or mesh, the returned nodes are checked against that domain.
"""
@inline supportnodes(bw::BasisWeight) = getfield(bw, :indices)[]
@inline function supportnodes(bw::BasisWeight, grid::Grid)
    supportnodes(bw, get_mesh(grid))
end
# An SpGrid always has a CartesianMesh, so the indices need no further mapping.
@inline function supportnodes(bw::BasisWeight, grid::SpGrid)
    inds = supportnodes(bw)
    spinds = get_spinds(grid)
    @boundscheck checkbounds(spinds, inds)
    @inbounds neighbors = view(spinds, inds)
    @debug @assert all(isactive, neighbors)
    neighbors
end

@inline function supportnodes(bw::BasisWeight, mesh::AbstractMesh)
    inds = supportnodes(bw)
    @boundscheck checkbounds(mesh, inds)
    inds
end

@inline supportnodes_storage(bw::BasisWeight) = getfield(bw, :indices)
@inline derivative_order(bw::BasisWeight) = getfield(bw, :order)

@generated function set_values!(bw::BasisWeight{B, Vals, Indices, Order{k}}, ip, vals::Tuple{Vararg{Any, N}}) where {B, Vals, Indices, k, N}
    N ≤ k+1 || return :(throw(DimensionMismatch("cannot write $N basis-value derivatives to Order($k) storage")))
    quote
        @_inline_meta
        @_propagate_inbounds_meta
        @nexprs $N i -> nodal_basis_values(bw, Order(i-1))[ip] = vals[i]
    end
end
@generated function set_values!(bw::BasisWeight{B, Vals, Indices, Order{k}}, vals::Tuple{Vararg{Any, N}}) where {B, Vals, Indices, k, N}
    N ≤ k+1 || return :(throw(DimensionMismatch("cannot write $N basis-value derivatives to Order($k) storage")))
    quote
        @_inline_meta
        @nexprs $N i -> copyto!(nodal_basis_values(bw, Order(i-1)), vals[i])
    end
end

function Base.show(io::IO, bw::BasisWeight)
    print(io, "BasisWeight: \n")
    print(io, "  Basis: ", basis(bw), "\n")
    print(io, "  Basis values: ")
    print(io, join(map(propertynames(bw)) do name
        string(name, "::", typeof(getproperty(bw, name)))
    end, ", "), "\n")
    print(io, "  Support nodes: ", supportnodes(bw))
end

# ---- BasisWeightArray ----

# A deferred `BasisWeightArray` keeps the shape and naming of a stored one --
# property order still gives the derivative order, element types still infer --
# while owning no memory. Reading one is a bug: the transfer lowering resolves
# deferred properties by evaluating the basis instead.
struct DeferredBasisValues{T, N} <: AbstractArray{T, N}
    dims::Dims{N}
end
DeferredBasisValues{T}(dims::Dims{N}) where {T, N} = DeferredBasisValues{T, N}(dims)
Base.size(A::DeferredBasisValues) = A.dims
Base.IndexStyle(::Type{<: DeferredBasisValues}) = IndexLinear()
Base.getindex(::DeferredBasisValues, ::Integer) =
    error("this basis value is not stored: the weights were built with `deferred=true`, so it is evaluated inside the transfer instead")
Base.similar(A::DeferredBasisValues{T}, ::Type{S}, dims::Dims) where {T, S} = DeferredBasisValues{S}(dims)

isdeferred(::DeferredBasisValues) = true
isdeferred(::AbstractArray) = false

# Owning no memory, a placeholder reports no backend and lets the surrounding
# arrays decide where the weights live.
KernelAbstractions.get_backend(::DeferredBasisValues) = nothing
Adapt.adapt_structure(to, A::DeferredBasisValues) = A

"""
    BasisWeightArray

Structure-of-arrays storage for multiple [`BasisWeight`](@ref)s.
Use [`generate_basis_weights`](@ref) to construct a `BasisWeightArray`.
"""
struct BasisWeightArray{B, Vals <: NamedTuple, Indices, ElType <: BasisWeight{B}, N, O <: Order, F} <: AbstractArray{ElType, N}
    basis::B
    vals::Vals
    indices::Indices
    order::O
    # A transfer resolves stored-versus-evaluated from the type it is handed, so
    # `select_weights` reads what `update!` recorded here to pick that type.
    deferring::F
end

function BasisWeightArray(basis::B, vals::Vals, indices::Indices, order::O, deferring::F=deferring_flag(basis)) where {B, Vals <: NamedTuple, N, Indices <: AbstractArray{<: Any, N}, O <: Order, F}
    ElType = Base._return_type(_getindex, Tuple{B, Vals, Indices, O, Vararg{Int, N}})
    BasisWeightArray{B, Vals, Indices, ElType, N, O, F}(basis, vals, indices, order, deferring)
end

# ---- deferral ----

# A type-level answer, so the branch in `select_weights` folds away entirely for
# weights that cannot defer.
can_defer(::Type{<: BasisWeightArray{B}}) where {B} = can_defer_basis(B)
can_defer(::Type) = false

deferring_flag(basis) = can_defer_basis(typeof(basis)) ? DeferralState() : nothing

# Which of the two reasons applies: the basis cannot be evaluated in a transfer,
# or the container has nowhere to record the choice.
@noinline _refuse_deferral(weights::BasisWeightArray) = check_deferred_basis(basis(weights))
@noinline _refuse_deferral(::AbstractArray{<: BasisWeight}) =
    error("update!: `deferred=true` needs a `BasisWeightArray`; a plain array of `BasisWeight`s has nowhere to record the choice, so a transfer would go on reading stored values.")

# The filter is what a basis correcting near boundaries needs and a transfer has
# no other way to learn. It stays untyped here and is read only on the host, in
# `select_weights`, which gives it a concrete type before anything reaches a kernel.
mutable struct DeferralState
    on::Bool
    filter::Any
end
DeferralState() = DeferralState(false, nothing)

# `update!` accepts any array of `BasisWeight`s; only `BasisWeightArray` has a
# slot, and the rest must reach the deferrability error rather than a FieldError.
@inline deferring_state(weights::BasisWeightArray) = getfield(weights, :deferring)
@inline deferring_state(::AbstractArray{<: BasisWeight}) = nothing

@inline _deferring(s::DeferralState) = s.on
@inline _deferring(::Nothing) = false
@inline _set_deferring!(s::DeferralState, on::Bool) = (s.on = on)
@inline _set_deferring!(::Nothing, ::Bool) = false
@inline _deferred_filter(s::DeferralState) = s.filter
@inline _deferred_filter(::Nothing) = nothing
@inline _deferred_filter(filter) = filter
# `Trues` means "every node", which the deferred path spells as `nothing` so a
# basis that ignores the filter carries nothing at all.
@inline _record_filter!(s::DeferralState, filter) = (s.filter = filter isa Trues ? nothing : filter)
@inline _record_filter!(::Nothing, filter) = nothing

# Never allocated, as opposed to allocated but currently bypassed. Only `update!`
# needs to tell those apart: it refills the second and has nothing to do for the first.
@inline storageless(weights::BasisWeightArray) = any(isdeferred, values(getfield(weights, :vals)))
@inline storageless(::AbstractArray{<: BasisWeight}) = false

# Weights carrying no flag have nothing to switch: they always defer or never can,
# and `update!` rejects `deferred=true` on the latter before it gets here.
set_deferring!(weights::BasisWeightArray, on::Bool) = (_set_deferring!(deferring_state(weights), on); weights)
set_deferring!(weights::AbstractArray{<: BasisWeight}, ::Bool) = weights

# `deferred` travels as a `Val` so the value arrays keep concrete types: a plain
# `Bool` widens the storage type to a union and degrades every downstream inference.
function _generate_basis_weights(::Type{Prop}, basis, mesh::AbstractMesh{dim}, dims::Dims; derivative::Order=Order(1), deferred=Val(false)) where {Prop <: NamedTuple, dim}
    _generate_basis_weights(Prop, basis, mesh, dims, derivative, asdeferredval(deferred))
end
function _generate_basis_weights(::Type{Prop}, basis, mesh::AbstractMesh{dim}, dims::Dims, derivative::Order, deferred::Val{L}) where {Prop <: NamedTuple, dim, L}
    L && check_deferred_basis(basis)
    template = allocate_basis_values(Prop, basis, Val(dim); derivative)
    njets = derivative_count(derivative)
    # Only the first `njets` properties are re-evaluated by a deferred array;
    # anything after them is caller-owned storage and is always allocated.
    vals = NamedTuple{keys(template)}(ntuple(Val(length(template))) do k
        v = template[k]
        sz = (size(v)..., dims...)
        L && k <= njets ? DeferredBasisValues{eltype(v)}(sz) : fill(zero(eltype(v)), sz)
    end)
    indices = _generate_supportnodes(basis, mesh, dims)
    BasisWeightArray(basis, vals, indices, derivative)
end
function _generate_basis_weights(::Type{T}, basis, mesh::AbstractMesh, dims::Dims; derivative::Order=Order(1), name=Val(:w), deferred=Val(false)) where {T}
    _generate_basis_weights(basis_property_type(T, name), basis, mesh, dims, derivative, asdeferredval(deferred))
end

@inline asdeferredval(::Val{L}) where {L} = Val(L::Bool)
@inline asdeferredval(deferred::Bool) = Val(deferred)

@inline derivative_count(::Order{k}) where {k} = k + 1

# A `Kernel` needs no per-particle state, its value at a node depending on nothing
# but that node. Bases fitting their values over the whole support do the fitting
# here rather than in `update!`.
@inline deferred_particle_state(::Order, basis, pt, mesh, window, filter) = nothing
@inline function deferred_node_jet(order::Order, basis, ::Nothing, pt, mesh, window, filter, ip)
    @_propagate_inbounds_meta
    nodal_basis_jet(order, basis, pt, mesh, window[ip])
end

# FEM and IGA shapes cannot be evaluated in a transfer: they also derive cell
# geometry that the equations read back from storage. One answer per basis, so
# none can be half-enabled.
can_defer_basis(::Type{<: Kernel}) = true
can_defer_basis(::Type) = false

check_deferred_basis(basis) =
    can_defer_basis(typeof(basis)) ? nothing :
    error("basis values can only be deferred for a basis a transfer can evaluate on its own, which $(nameof(typeof(basis))) is not: it also derives cell geometry that the equations read back from storage.")

_generate_supportnodes(basis, mesh::CartesianMesh, dims) = map(_ -> initial_supportnodes(basis, mesh), CartesianIndices(dims))

# FEM/IGA: every quadrature point of a cell shares that cell's support.
mutable struct CellSupportMatrix{T, V <: AbstractVector{T}} <: AbstractMatrix{T}
    const dims::Dims{2}
    cellsupports::V
end

function CellSupportMatrix(supports::V, nq::Int, ncells::Int) where {T, V <: AbstractVector{T}}
    ncells == length(supports) || throw(DimensionMismatch("the second basis-weight dimension must equal the number of cells"))
    CellSupportMatrix{T, V}((nq, ncells), supports)
end

Base.size(A::CellSupportMatrix) = A.dims
Base.IndexStyle(::Type{<: CellSupportMatrix}) = IndexCartesian()
cellsupports(A::CellSupportMatrix) = getfield(A, :cellsupports)
@inline function Base.getindex(A::CellSupportMatrix, q::Int, cell::Int)
    @boundscheck checkbounds(A, q, cell)
    @inbounds cellsupports(A)[cell]
end

function set_cellsupports!(A::CellSupportMatrix{T, V}, supports::V) where {T, V}
    length(supports) == size(A, 2) || throw(DimensionMismatch("the number of cells must match the second basis-weight dimension"))
    setfield!(A, :cellsupports, supports)
    A
end

_generate_supportnodes(::Shape, mesh::FEMesh, dims::Dims{2}) = CellSupportMatrix(cellsupports(mesh), dims...)
_generate_supportnodes(::Shape, ::FEMesh, ::Dims) = throw(DimensionMismatch("FEM basis weights must have dimensions (quadrature points, cells)"))

function _generate_cell_supportnodes(mesh, dims::Dims{2})
    dims[2] == ncells(mesh) || throw(DimensionMismatch("the second basis-weight dimension must equal the number of cells"))
    CellSupportMatrix(map(cell -> supportnodes(mesh, cell), cells(mesh)), dims...)
end

_todims(x::Tuple{Vararg{Int}}) = x
_todims(x::Vararg{Int}) = x

"""
    generate_basis_weights([T,] ::Basis, mesh, dims...; derivative=Order(1), name=Val(:w))
    generate_basis_weights([T,] ::FEMesh, dims...; derivative=Order(1), name=Val(:w))
    generate_basis_weights(Prop, ::Basis, mesh, dims...; derivative=Order(1))
    generate_basis_weights(Prop, ::FEMesh, dims...; derivative=Order(1))

Generate an array of [`BasisWeight`](@ref)s for `basis` on `mesh`.
For `FEMesh`, the mesh cell shape is used as the basis.

In the `Prop` forms, `Prop` follows the same rules as in
`BasisWeight(Prop, basis, mesh)`: its first field defines the basis value name
and scalar type, generated derivative fields follow it, and its remaining
fields are appended as custom storage.

Passing `deferred=true` allocates no storage for the basis values and evaluates
them inside each transfer instead. See the manual section on deferred basis
weights for what that trades and which bases allow it.
"""
function generate_basis_weights end

generate_basis_weights(::Type{T}, basis::Basis, mesh::CartesianMesh, dims...; kwargs...) where {T} = _generate_basis_weights(T, basis, mesh, _todims(dims...); kwargs...)
generate_basis_weights(basis::Basis, mesh::CartesianMesh, dims...; kwargs...) = _generate_basis_weights(Float64, basis, mesh, _todims(dims...); kwargs...)

generate_basis_weights(::Type{T}, mesh::FEMesh, dims...; kwargs...) where {T} = _generate_basis_weights(T, basis(mesh), mesh, _todims(dims...); kwargs...)
generate_basis_weights(mesh::FEMesh, dims...; kwargs...) = _generate_basis_weights(Float64, basis(mesh), mesh, _todims(dims...); kwargs...)

Base.size(x::BasisWeightArray) = size(getfield(x, :indices))

@inline function Base.view(x::BasisWeightArray, I...)
    indices = view(getfield(x, :indices), I...)
    vals = map(vals -> viewcol(vals, Val(ndims(x)), I...), getfield(x, :vals))
    # The slot is shared, not copied: a view taken once outside the time loop
    # then follows the parent's per-step `update!(...; deferred=true/false)`.
    BasisWeightArray(getfield(x, :basis), vals, indices, derivative_order(x), getfield(x, :deferring))
end

Base.propertynames(x::BasisWeightArray) = propertynames(getfield(x, :vals))
@inline function Base.getproperty(x::BasisWeightArray, name::Symbol)
    getproperty(getfield(x, :vals), name)
end

@inline basis(x::BasisWeightArray) = getfield(x, :basis)
@inline derivative_order(x::BasisWeightArray) = getfield(x, :order)
@inline supportnodes_storage(x::BasisWeightArray) = getfield(x, :indices)

@inline function Base.getindex(x::BasisWeightArray{<: Any, <: Any, <: Any, <: Any, N}, I::Vararg{Integer, N}) where {N}
    @boundscheck checkbounds(x, I...)
    @inbounds _getindex(getfield(x, :basis), getfield(x, :vals), getfield(x, :indices), derivative_order(x), I...)
end
@generated function _getindex(basis, vals::NamedTuple{names}, indices::AbstractArray{<: Any, N}, order::Order, I::Vararg{Integer, N}) where {names, N}
    exps = [:(viewcol(vals.$name, I...)) for name in names]
    quote
        @_inline_meta
        @_propagate_inbounds_meta
        BasisWeight(basis, NamedTuple{names}(tuple($(exps...))), view_supportnodes(indices, I...), order)
    end
end

# When the storage is itself an integer-vector view (a subset of particles, such
# as a constitutive group), composing range views would eagerly reindex
# `parent_index[i:i]`, which allocates inside GPU kernels; hence the scalar read.
@inline view_supportnodes(indices::AbstractArray{<: Any, N}, I::Vararg{Integer, N}) where {N} = view(indices, map(:, I, I)...)
@inline function view_supportnodes(indices::SubArray{<: Any, 1, <: AbstractVector, <: Tuple{AbstractVector{<: Integer}}}, i::Integer)
    @_propagate_inbounds_meta
    j = parentindices(indices)[1][i]
    view(parent(indices), j:j)
end

@inline function viewcol(A::AbstractArray, I::Vararg{Integer, N}) where {N}
    viewcol(A, Val(N), I...)
end
@inline function viewcol(A::AbstractArray, ::Val{N}, I...) where {N}
    colons = nfill(:, Val(ndims(A)-N))
    @boundscheck checkbounds(A, colons..., I...)
    @inbounds view(A, colons..., I...)
end
# A placeholder must not become a `SubArray`: `deferred_order` looks for
# `DeferredBasisValues` in the field type, so wrapping it would make a view of
# deferred weights read storage that was never filled.
@inline function viewcol(A::DeferredBasisValues{T}, ::Val{N}, I...) where {T, N}
    colons = nfill(:, Val(ndims(A)-N))
    @boundscheck checkbounds(A, colons..., I...)
    DeferredBasisValues{T}(map(length, Base.index_shape(Base.to_indices(A, (colons..., I...))...)))
end

function _show_basis_weight_array(io::IO, weights::BasisWeightArray)
    bw = first(weights)
    print(io, Base.dims2string(size(weights)), " ", ndims(weights)==1 ? "BasisWeightVector" : "BasisWeightArray", ": \n")
    print(io, "  Basis: ", basis(weights), "\n")
    print(io, "  Basis values: ", join(propertynames(bw), ", "))
end

Base.show(io::IO, ::MIME"text/plain", weights::BasisWeightArray) = _show_basis_weight_array(io, weights)
Base.show(io::IO, weights::BasisWeightArray) = _show_basis_weight_array(io, weights)

# ---- update! ----

@inline function alltrue(A::AbstractArray{Bool}, indices::CartesianIndices)
    @debug checkbounds(A, indices)
    @inbounds for i in indices
        A[i] || return false
    end
    true
end
# A deferred transfer spells "no filter" as `nothing`.
@inline filterpasses(::Nothing, i) = true
@inline filterpasses(A::AbstractArray{Bool}, i) = (@_propagate_inbounds_meta; A[i])

@inline alltrue(::Nothing, ::CartesianIndices) = true
@inline function alltrue(A::Trues, indices::CartesianIndices)
    @debug checkbounds(A, indices)
    true
end
@inline has_full_support(bw::BasisWeight, indices) = size(nodal_basis_values(bw, Order(0))) == size(indices)
@inline has_full_support(bw::BasisWeight, indices, filter::AbstractArray{Bool}) = has_full_support(bw, indices) && alltrue(filter, indices)

@inline function require_filtered_update_support(b)
    supports_filtered_updates(b) || throw(ArgumentError("$(typeof(b)) does not support filtered updates"))
    nothing
end
@inline supports_filtered_updates(::Any) = false
@inline require_filtered_update_support(weights::BasisWeightArray, ::Integer) = require_filtered_update_support(basis(weights))
@inline function require_filtered_update_support(weights::AbstractArray{<: BasisWeight}, n::Integer)
    @inbounds for p in 1:n
        require_filtered_update_support(basis(weights[p]))
    end
end

@inline update_basis_values!(bw::BasisWeight, kernel::Kernel, pt, mesh::CartesianMesh) = update_basis_values_nodewise!(bw, kernel, pt, mesh)
@inline function update_basis_values_nodewise!(bw::BasisWeight, kernel::Kernel, pt, mesh::CartesianMesh)
    indices = supportnodes(bw)
    order = derivative_order(bw)
    @inbounds for ip in eachindex(indices)
        set_values!(bw, ip, nodal_basis_jet(order, kernel, pt, mesh, indices[ip]))
    end
    bw
end

@inline function update!(bw::BasisWeight, pt, mesh::AbstractMesh)
    update_basis_weight!(bw, pt, mesh)
    bw
end
@inline function update_basis_weight!(bw::BasisWeight, pt, mesh::AbstractMesh)
    b = basis(bw)
    supportnodes_storage(bw)[] = supportnodes(b, pt, mesh)
    update_basis_values!(bw, b, pt, mesh)
end
@inline function update!(bw::BasisWeight, pt, mesh::AbstractMesh, filter::AbstractArray{Bool})
    @assert size(mesh) == size(filter)
    require_filtered_update_support(basis(bw))
    update_basis_weight!(bw, pt, mesh, filter)
    bw
end
@inline update!(bw::BasisWeight, pt, mesh::AbstractMesh, ::Trues) = update!(bw, pt, mesh)
@inline function update_basis_weight!(bw::BasisWeight, pt, mesh::AbstractMesh, filter::AbstractArray{Bool})
    b = basis(bw)
    supportnodes_storage(bw)[] = supportnodes(b, pt, mesh)
    update_basis_values!(bw, b, pt, mesh, filter)
end

"""
    Tesserae.isdeferred(weights)

Whether transfers evaluate `weights`' basis values instead of reading stored
ones. True both for weights built with `generate_basis_weights(...; deferred=true)`,
which store nothing, and for stored weights switched over with
`update!(..., deferred=true)`.
"""
isdeferred(weights::BasisWeightArray) = storageless(weights) || _deferring(deferring_state(weights))
@inline deferred_filter(weights::BasisWeightArray) = _deferred_filter(deferring_state(weights))

# A deferred twin sharing everything with `weights` and owning no storage, so
# `select_weights` can build one per transfer for free. Internal: this is how
# `update!(...; deferred=true)` reaches the type.
function as_deferred(weights::BasisWeightArray, filter = deferred_filter(weights))
    b = basis(weights)
    order = derivative_order(weights)
    njets = derivative_count(order)
    storageless(weights) && return BasisWeightArray(b, getfield(weights, :vals), getfield(weights, :indices), order, filter)
    stored = getfield(weights, :vals)
    # Mirrors `_generate_basis_weights`: only the first `njets` properties are
    # re-evaluated, the caller-owned storage after them being shared as-is.
    vals = NamedTuple{keys(stored)}(ntuple(Val(length(stored))) do k
        v = stored[k]
        k <= njets ? DeferredBasisValues{eltype(v)}(size(v)) : v
    end)
    BasisWeightArray(b, vals, getfield(weights, :indices), order, filter)
end

# See the note on `@Const` and nested containers in src/Transfer/gpu.jl.
@kernel function gpukernel_update_weight(weights, particles, mesh, filter)
    p = @index(Global)
    update!(weights[p], LazyRow(particles, p), mesh, filter)
end

"""
    update!(weights, particles, mesh)

Updates each element in `weights` using particle data and the background `mesh`.
Automatically dispatches to CPU or GPU backend with appropriate parallelization.

This is functionally equivalent to:

```julia
for p in eachindex(particles)
    update!(weights[p], LazyRow(particles, p), mesh)
end
```

where [`LazyRow`](https://juliaarrays.github.io/StructArrays.jl/stable/#Lazy-row-iteration) is provided in [StructArrays.jl](https://github.com/JuliaArrays/StructArrays.jl).

On weights built with `generate_basis_weights(...; deferred=true)` there is no storage
to fill, so this is a no-op and a loop written for stored weights runs unchanged.
On weights that do store their values, `deferred=true` skips the fill and tells
transfers to evaluate instead of read, until a later `update!` without it refills
them. It is the per-step form of the same choice, and it errors on a basis that
cannot defer rather than silently doing nothing.
"""
function update!(weights::AbstractArray{<: BasisWeight}, particles::StructArray, mesh::AbstractMesh,
                 filter::AbstractArray{Bool}=Trues(size(mesh)); deferred::Bool=storageless(weights))
    # Recorded before the early return below: weights that store nothing still
    # need the filter, it being their transfers that do the correcting.
    if deferred || storageless(weights)
        _record_filter!(deferring_state(weights), filter)
    end
    if storageless(weights)
        deferred || error("update!: these weights were built with `deferred=true` and have no storage for basis values, so they cannot be materialized. Build them without `deferred=true` to store the values.")
        return weights
    end

    if deferred
        can_defer(typeof(weights)) || _refuse_deferral(weights)
        return set_deferring!(weights, true)
    end
    set_deferring!(weights, false)

    n = length(particles)
    @assert length(weights) ≥ n
    @assert size(mesh) == size(filter)
    if !(filter isa Trues) && n != 0
        require_filtered_update_support(weights, n)
    end

    backend = get_backend(weights)
    @assert get_backend(particles) == get_backend(mesh) == backend
    @assert filter isa Trues || get_backend(filter) == backend

    if backend isa CPU
        @threaded for p in 1:n
            @inbounds update!(weights[p], LazyRow(particles, p), mesh, filter)
        end
    else
        kernel = gpukernel_update_weight(backend)
        kernel(weights, particles, mesh, filter; ndrange=n)
    end
    weights
end
