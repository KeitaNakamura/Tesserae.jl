# Deferred basis values and evaluation policy

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
Base.reshape(A::DeferredBasisValues{T}, dims::Dims) where {T} =
    DeferredBasisValues{T}(size(reshape(Base.OneTo(length(A)), dims)))

isdeferred(::DeferredBasisValues) = true
isdeferred(::AbstractArray) = false

# Owning no memory, a placeholder reports no backend and lets the surrounding
# arrays decide where the weights live.
KernelAbstractions.get_backend(::DeferredBasisValues) = nothing
Adapt.adapt_structure(to, A::DeferredBasisValues) = A

# ---- deferral ----

deferring_flag(basis) = can_defer_basis(typeof(basis)) ? DeferralState() : nothing

# The filter is what a basis correcting near boundaries needs and a transfer has
# no other way to learn. Its concrete type is resolved only on the host, in
# `select_weights`, which gives it a concrete type before anything reaches a kernel.
mutable struct DeferralState
    on::Bool
    filter::Union{Nothing,AbstractArray{Bool}}
end
DeferralState() = DeferralState(false, nothing)

# `update!` accepts any array of `BasisWeight`s; only `BasisWeightArray` has a
# slot, and the rest must reach the deferrability error rather than a FieldError.
@inline deferring_state(weights::BasisWeightArray) = getfield(weights, :deferring)
@inline deferring_state(::AbstractArray{<: BasisWeight}) = nothing

# `Trues` means "every node", which the deferred path spells as `nothing` so a
# basis that ignores the filter carries nothing at all.
@inline normalized_filter(filter) = filter isa Trues ? nothing : filter

# Never allocated, as opposed to allocated but currently bypassed. Only `update!`
# needs to tell those apart: it refills the second and has nothing to do for the first.
@inline storageless(weights::BasisWeightArray) = any(isdeferred, values(getfield(weights, :vals)))
@inline storageless(::AbstractArray{<: BasisWeight}) = false

@inline asdeferredval(::Val{L}) where {L} = Val(L::Bool)
@inline asdeferredval(deferred::Bool) = Val(deferred)

# The default kernel evaluates directly at each node. Bases that can reuse
# axis values or a fit over the support build that state once per particle here.
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

# Which of the two reasons applies: the basis cannot be evaluated in a transfer,
# or the container has nowhere to record the choice.
check_deferred_support(weights::BasisWeightArray) = check_deferred_support(basis(weights))
@noinline check_deferred_support(::AbstractArray{<: BasisWeight}) =
    error("update!: `deferred=true` needs a `BasisWeightArray`; a plain array of `BasisWeight`s has nowhere to record the choice, so a transfer would go on reading stored values.")

check_deferred_support(basis) =
    can_defer_basis(typeof(basis)) ? nothing :
    error("basis values can only be deferred for a basis a transfer can evaluate on its own, which $(nameof(typeof(basis))) is not: it also derives cell geometry that the equations read back from storage.")

# A placeholder must not become a `SubArray`: `deferred_order` looks for
# `DeferredBasisValues` in the field type, so wrapping it would make a view of
# deferred weights read storage that was never filled.
@inline function viewcol(A::DeferredBasisValues{T}, ::Val{N}, I...) where {T, N}
    colons = nfill(:, Val(ndims(A)-N))
    @boundscheck checkbounds(A, colons..., I...)
    DeferredBasisValues{T}(map(length, Base.index_shape(Base.to_indices(A, (colons..., I...))...)))
end

"""
    Tesserae.isdeferred(weights)

Whether transfers evaluate `weights`' basis values instead of reading stored
ones. True both for weights built with `generate_basis_weights(...; deferred=true)`,
which store nothing, and for stored weights switched over with
`update!(..., deferred=true)`.
"""
@inline function isdeferred(weights::Union{BasisWeightArray,BasisWeightView})
    state = deferring_state(weights)
    storageless(weights) || (state !== nothing && state.on)
end
@inline function deferred_filter(weights::Union{BasisWeightArray,BasisWeightView})
    state = deferring_state(weights)
    state === nothing ? nothing : state.filter
end

@inline deferring_state(weights::BasisWeightView) = deferring_state(basis_weight_parent(weights))
@inline storageless(weights::BasisWeightView) = storageless(basis_weight_parent(weights))
