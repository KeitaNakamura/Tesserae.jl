# Basis-weight storage and updates

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

zero_basis_value(::Type{Vec{dim, T}}, ::Order{0}) where {dim, T} = zero(T)
zero_basis_value(::Type{Vec{dim, T}}, ::Order{1}) where {dim, T} = zero(Vec{dim, T})
zero_basis_value(::Type{Vec{dim, T}}, ::Order{k}) where {dim, T, k} = zero(Tensor{Tuple{@Symmetry{ntuple(i->dim, k)...}}, T})
basis_value_name(::Order{0}, ::Val{name}) where {name} = name
for (k, nabla) in enumerate((:∇, :∇², :∇³, :∇⁴, :∇⁵, :∇⁶, :∇⁷, :∇⁸, :∇⁹))
    @eval begin
        basis_value_name(::Order{$k}, ::Val{name}) where {name} = Symbol($(QuoteNode(nabla)), name)
    end
end

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
@inline function nodal_basis_values(bw::BasisWeight{B, Vals, Indices, Order{n}}, ::Order{k}) where {B, Vals, Indices, n, k}
    k ≤ n || throw(ArgumentError("basis weight stores derivatives through Order($n), got Order($k)"))
    getfield(bw, :vals)[k+1]
end

@inline scalartype(bw::BasisWeight) = eltype(nodal_basis_values(bw, Order(0)))

"""
    basis(mesh::FEMesh)
    basis(mesh::IGAMesh)
    basis(weight)

Return the basis associated with a mesh or basis-weight storage.
"""
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

# `deferred` travels as a `Val` so the value arrays keep concrete types: a plain
# `Bool` widens the storage type to a union and degrades every downstream inference.
function _generate_basis_weights(::Type{Prop}, basis, mesh::AbstractMesh{dim}, dims::Dims; derivative::Order=Order(1), deferred=Val(false)) where {Prop <: NamedTuple, dim}
    _generate_basis_weights(Prop, basis, mesh, dims, derivative, asdeferredval(deferred))
end
function _generate_basis_weights(::Type{Prop}, basis, mesh::AbstractMesh{dim}, dims::Dims, derivative::Order, deferred::Val{L}) where {Prop <: NamedTuple, dim, L}
    L && check_deferred_support(basis)
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

_generate_supportnodes(basis, mesh::CartesianMesh, dims) = map(_ -> initial_supportnodes(basis, mesh), CartesianIndices(dims))

_todims(x::Tuple{Vararg{Int}}) = x
_todims(x::Vararg{Int}) = x

"""
    generate_basis_weights([T,] ::Basis, mesh, dims...; derivative=Order(1), name=Val(:w))
    generate_basis_weights([T,] ::FEMesh, dims...; derivative=Order(1), name=Val(:w))
    generate_basis_weights([T,] ::IGAMesh, dims...; derivative=Order(1), name=Val(:w))
    generate_basis_weights(Prop, ::Basis, mesh, dims...; derivative=Order(1))
    generate_basis_weights(Prop, ::FEMesh, dims...; derivative=Order(1))

Generate an array of [`BasisWeight`](@ref)s for `basis` on `mesh`.
For `FEMesh`, the mesh cell shape is used as the basis; for `IGAMesh`, the
basis is taken from the patch degrees. The `Prop` forms cover `IGAMesh` too.
FEM and IGA weights support `Order(0)` and `Order(1)` only; domain updates
require `Order(1)`.

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

Base.size(x::BasisWeightArray) = size(getfield(x, :indices))

# Native views keep their parent's element type and row handles. The SoA
# projection is prepared only when an update or transfer needs bulk storage.
const BasisWeightView = Adapt.WrappedSubArray{<:BasisWeight,<:Any,<:BasisWeightArray}

@inline function Base.view(x::Union{BasisWeightArray,BasisWeightView}, I...)
    J = basis_weight_indices(x, I)
    invoke(view, Tuple{AbstractArray,Vararg{Any}}, x, J...)
end

basis_weight_storage(x::BasisWeightArray) = x
function basis_weight_storage(x::SubArray)
    p = basis_weight_storage(parent(x))
    I = parentindices(x)
    # The native view has already checked these particle indices.
    vals = map(a -> (@inbounds viewcol(a, Val(ndims(p)), I...)), getfield(p, :vals))
    indices = @inbounds view(supportnodes_storage(p), I...)
    BasisWeightArray(basis(p), vals, indices,
                     derivative_order(p), deferring_state(p))
end
function basis_weight_storage(x::Base.ReshapedArray)
    p = basis_weight_storage(parent(x))
    vals = map(getfield(p, :vals)) do a
        leading = ntuple(i -> size(a,i), Val(ndims(a)-ndims(p)))
        reshape(a, (leading..., size(x)...))
    end
    BasisWeightArray(basis(p), vals, reshape(supportnodes_storage(p), size(x)),
                     derivative_order(p), deferring_state(p))
end

basis_weight_parent(x::BasisWeightArray) = x
basis_weight_parent(x::Union{SubArray,Base.ReshapedArray}) = basis_weight_parent(parent(x))
Base.propertynames(x::BasisWeightView) = propertynames(basis_weight_parent(x))
@inline function Base.getproperty(x::BasisWeightView, name::Symbol)
    hasfield(typeof(x), name) && return getfield(x, name)
    getproperty(basis_weight_storage(x), name)
end
@inline basis(x::BasisWeightView) = basis(basis_weight_parent(x))
@inline derivative_order(x::BasisWeightView) = derivative_order(basis_weight_parent(x))
@inline supportnodes_storage(x::BasisWeightView) = supportnodes_storage(basis_weight_storage(x))

@inline function basis_weight_indices(x::Union{BasisWeightArray,BasisWeightView}, I)
    @_propagate_inbounds_meta
    # Validate mask axes before a backend turns masks into integer positions.
    @boundscheck if any(i -> i isa AbstractArray{Bool}, I)
        shapes = map(I) do i
            if i isa AbstractArray{Bool}
                axes(i) == map(Base.OneTo, size(i)) || throw(BoundsError(x, I))
                # Only axes matter here. GPU index reducers can otherwise
                # interpret Boolean values as the integer indices 0 and 1.
                Trues(size(i))
            else
                i
            end
        end
        checkbounds(x, shapes...)
    end
    Base.to_indices(supportnodes_storage(x), I)
end

Base.propertynames(x::BasisWeightArray) = propertynames(getfield(x, :vals))
@inline function Base.getproperty(x::BasisWeightArray, name::Symbol)
    getproperty(getfield(x, :vals), name)
end

@inline basis(x::BasisWeightArray) = getfield(x, :basis)
@inline derivative_order(x::BasisWeightArray) = getfield(x, :order)
@inline supportnodes_storage(x::BasisWeightArray) = getfield(x, :indices)

@inline function Base.getindex(x::BasisWeightArray{B,Vals,Indices,ElType,N}, I::Vararg{Integer,N}) where {B,Vals,Indices,ElType,N}
    J = map(Base.to_index, I)
    @boundscheck checkbounds(x, J...)
    @inbounds _getindex(basis(x), getfield(x, :vals), supportnodes_storage(x), derivative_order(x), J...)
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
function _show_basis_weight_array(io::IO, weights::BasisWeightArray)
    print(io, Base.dims2string(size(weights)), " ", ndims(weights)==1 ? "BasisWeightVector" : "BasisWeightArray", ": \n")
    print(io, "  Basis: ", basis(weights), "\n")
    print(io, "  Basis values: ", join(propertynames(weights), ", "))
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

# See the note on `@Const` and nested containers in src/Transfer/gpu.jl.
@kernel function gpukernel_update_weight(weights, particles, mesh, filter)
    p = @index(Global)
    update!(weights[p], LazyRows(particles)[p], mesh, filter)
end

"""
    update!(weights, particles, mesh)

Updates each element in `weights` using particle data and the background `mesh`.
Automatically dispatches to CPU or GPU backend with appropriate parallelization.

This is functionally equivalent to:

```julia
for p in eachindex(particles)
    update!(weights[p], LazyRows(particles)[p], mesh)
end
```

where [`LazyRows`](https://juliaarrays.github.io/StructArrays.jl/stable/#Lazy-row-iteration) is provided in [StructArrays.jl](https://github.com/JuliaArrays/StructArrays.jl).

On weights built with `generate_basis_weights(...; deferred=true)` there is no storage
to fill, but the arguments are still validated and the filter is recorded.
On weights that do store their values, `deferred=true` skips the fill and tells
transfers to evaluate instead of read, until a later `update!` without it refills
them. It is the per-step form of the same choice, and it errors on a basis that
cannot defer rather than silently doing nothing.

A `BasisWeightArray` view inherits the parent's current evaluation mode and
filter when they are omitted. It cannot change the mode. In stored mode an
explicit filter applies only to the view's values, without changing the parent's
filter. In deferred mode an explicit filter must be the same object as the
parent's filter; change that filter by updating the parent weights instead.
"""
function update!(weights::AbstractArray{<: BasisWeight}, particles::StructArray, mesh::AbstractMesh,
                 filter::AbstractArray{Bool}=Trues(size(mesh)); deferred::Bool=storageless(weights))
    _update_basis_weights!(weights, particles, mesh, filter, deferred)
    state = deferring_state(weights)
    if state !== nothing
        state.on = deferred
        state.filter = normalized_filter(filter)
    end
    weights
end

# Shared validation and filling; only the owner's public update records policy.
function _update_basis_weights!(weights, particles, mesh, filter, deferred::Bool)
    if storageless(weights)
        deferred || error("update!: these weights were built with `deferred=true` and have no storage for basis values, so they cannot be materialized. Build them without `deferred=true` to store the values.")
    elseif deferred
        check_deferred_support(weights)
    end

    n = length(particles)
    @assert length(weights) ≥ n
    @assert size(mesh) == size(filter)
    if !(filter isa Trues) && n != 0
        require_filtered_update_support(weights, n)
    end

    backend = get_backend(weights)
    @assert get_backend(particles) == get_backend(mesh) == backend
    @assert filter isa Trues || get_backend(filter) == backend

    # Validate every request before writing a partial fill.
    if !deferred
        if backend isa CPU
            @threaded for p in 1:n
                @inbounds update!(weights[p], LazyRows(particles)[p], mesh, filter)
            end
        elseif n != 0
            kernel = gpukernel_update_weight(backend)
            kernel(weights, particles, mesh, filter; ndrange=n)
        end
    end
    weights
end

# Native views retain the owner's policy; bulk operations use a typed projection.
# Keep this wrapper small so a runtime-typed filter does not rebox the returned view.
function update!(weights::BasisWeightView, particles::StructArray, mesh::AbstractMesh,
                 filter::AbstractArray{Bool}=(@something deferred_filter(weights) Trues(size(mesh)));
                 deferred::Bool=isdeferred(weights))
    _update_basis_weights!(weights, particles, mesh, filter, deferred)
    weights
end
function _update_basis_weights!(weights::BasisWeightView, particles, mesh, filter, deferred::Bool)
    if deferring_state(weights) !== nothing
        deferred == isdeferred(weights) ||
            throw(ArgumentError("update!: a basis-weight view cannot change the evaluation mode; update the parent weights instead"))
        if deferred && normalized_filter(filter) !== deferred_filter(weights)
            throw(ArgumentError("update!: a deferred basis-weight view must use the parent's filter; omit the filter or pass the same filter object, and update the parent weights to change it"))
        end
    end
    _update_basis_weights!(basis_weight_storage(weights), particles, mesh, filter, deferred)
    nothing
end
