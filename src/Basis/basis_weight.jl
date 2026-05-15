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

#=
To create a new basis, following methods need to be implemented.
* Tesserae.create_property(::Type{Vec{dim, T}}, basis; kwargs...) -> NamedTuple
* Tesserae.initial_supportnodes(basis, mesh)
* Tesserae.update_property!(bw::BasisWeight, basis, pt, mesh)
=#

initial_supportnodes(::Basis, ::CartesianMesh{dim}) where {dim} = EmptyCartesianIndices(Val(dim))
initial_supportnodes(shape::Shape, mesh::UnstructuredMesh) = zero(SVector{nlocalnodes(shape), Int})

propsize(basis::Basis, ::Val{dim}) where{dim} = nfill(kernel_support(basis), Val(dim))
propsize(shape::Shape, ::Val)  = (nlocalnodes(shape),)
function create_property(::Type{Vec{dim, T}}, basis; derivative::Order{k}=Order(1), name=nothing) where {dim, T, k}
    map(Array, create_property(MArray, Vec{dim, T}, basis; derivative, name))
end
@generated function create_property(::Type{MArray}, ::Type{Vec{dim, T}}, basis; derivative::Order{k}=Order(1), name=nothing) where {dim, T, k}
    quote
        arrdims = propsize(basis, Val(dim))
        names = @ntuple $(k+1) i -> create_name(Order(i-1), name)
        vals = @ntuple $(k+1) i -> fill(zero(create_elval(Vec{dim, T}, Order(i-1))), MArray{Tuple{arrdims...}})
        NamedTuple{names}(vals)
    end
end

create_elval(::Type{Vec{dim, T}}, ::Order{0}) where {dim, T} = zero(T)
create_elval(::Type{Vec{dim, T}}, ::Order{1}) where {dim, T} = zero(Vec{dim, T})
create_elval(::Type{Vec{dim, T}}, ::Order{k}) where {dim, T, k} = zero(Tensor{Tuple{@Symmetry{ntuple(i->dim, k)...}}, T})
create_name(::Order{0}, ::Val{name}) where {name} = name
create_name(::Order{0}, ::Nothing) = :w
for (k, nabla) in enumerate((:∇, :∇², :∇³, :∇⁴, :∇⁵, :∇⁶, :∇⁷, :∇⁸, :∇⁹))
    @eval begin
        create_name(::Order{$k}, ::Val{name}) where {name} = Symbol($(QuoteNode(nabla)), name)
        create_name(::Order{$k}, ::Nothing) = $(QuoteNode(Symbol(nabla, :w)))
    end
end

@inline function prod_each_dimension(::Order{0}, vals::Vararg{Tuple, dim}) where {dim}
    tuple_otimes(ntuple(d -> vals[d][1], Val(dim)))
end
@generated function prod_each_dimension(::Order{k}, vals::Vararg{Tuple, dim}) where {k, dim}
    if k == 1
        TT = Vec{dim}
    else
        TT = Tensor{Tuple{@Symmetry{fill(dim,k)...}}}
    end
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

"""
    BasisWeight([T,] basis, mesh)

`BasisWeight` stores basis function values and their spatial derivatives.

```jldoctest
julia> mesh = CartesianMesh(1.0, (0,5), (0,5));

julia> xₚ = Vec(2.2, 3.4); # particle position

julia> bw = BasisWeight(BSpline(Quadratic()), mesh);

julia> update!(bw, xₚ, mesh) # update `bw` at position `xₚ` in `mesh`
BasisWeight:
  Basis: BSpline(Quadratic())
  Property names: w::Matrix{Float64}, ∇w::Matrix{Vec{2, Float64}}
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
struct BasisWeight{B, Prop <: NamedTuple, Indices <: AbstractArray{<: Any}}
    basis::B
    prop::Prop
    indices::Indices
end

# AbstractMesh
function _basis_weight(::Type{T}, basis, mesh::AbstractMesh{dim}; kwargs...) where {T, dim}
    prop = create_property(Vec{dim, T}, basis; kwargs...)
    indices = initial_supportnodes(basis, mesh)
    BasisWeight(basis, prop, fill(indices))
end

# CartesianMesh
BasisWeight(::Type{T}, basis::Basis, mesh::CartesianMesh; kwargs...) where {T} = _basis_weight(T, basis, mesh; kwargs...)
BasisWeight(basis::Basis, mesh::CartesianMesh; kwargs...) = _basis_weight(Float64, basis, mesh; kwargs...)

# UnstructuredMesh
BasisWeight(::Type{T}, mesh::UnstructuredMesh; kwargs...) where {T} = _basis_weight(T, cellshape(mesh), mesh; kwargs...)
BasisWeight(mesh::UnstructuredMesh; kwargs...) = BasisWeight(Float64, mesh; kwargs...)

Base.propertynames(bw::BasisWeight) = propertynames(getfield(bw, :prop))
@inline function Base.getproperty(bw::BasisWeight, name::Symbol)
    getproperty(getfield(bw, :prop), name)
end
@inline function Base.values(bw::BasisWeight, i::Int)
    getfield(bw, :prop)[i]
end

@inline scalartype(bw::BasisWeight) = eltype(values(bw, 1))

"""
    basis(weight)

Return the basis object used by a [`BasisWeight`](@ref) or [`BasisWeightArray`](@ref).
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
@inline function supportnodes(bw::BasisWeight, mesh::CartesianMesh)
    inds = supportnodes(bw)
    @boundscheck checkbounds(mesh, inds)
    inds
end
# SpGrid always use CartesianMesh
@inline function supportnodes(bw::BasisWeight, grid::SpGrid)
    inds = supportnodes(bw)
    spinds = get_spinds(grid)
    @boundscheck checkbounds(spinds, inds)
    @inbounds neighbors = view(spinds, inds)
    @debug @assert all(isactive, neighbors)
    neighbors
end

@inline function supportnodes(bw::BasisWeight, mesh::UnstructuredMesh)
    inds = supportnodes(bw)
    @boundscheck checkbounds(mesh, inds)
    inds
end

@inline supportnodes_storage(bw::BasisWeight) = getfield(bw, :indices)

@inline function derivative_order(bw::BasisWeight)
    @debug check_weight_prop(bw)
    k = length(propertynames(bw)) - 1
    Order(k)
end
@inline function check_weight_prop(bw::BasisWeight)
    k = length(propertynames(bw)) - 1
    _check_weight_prop(bw, Val(k))
end
@generated function _check_weight_prop(bw::BasisWeight, ::Val{k}) where {k}
    quote
        @_inline_meta
        @assert @nall $(k+1) i -> create_name(Order(i-1), Val(propertynames(bw)[1])) === propertynames(bw)[i]
    end
end

@generated function set_values!(bw::BasisWeight, ip, vals::Tuple{Vararg{Any, N}}) where {N}
    quote
        @_inline_meta
        @_propagate_inbounds_meta
        @nexprs $N i -> values(bw, i)[ip] = vals[i]
    end
end
@generated function set_values!(bw::BasisWeight, vals::Tuple{Vararg{Any, N}}) where {N}
    quote
        @_inline_meta
        @nexprs $N i -> copyto!(values(bw, i), vals[i])
    end
end

function Base.show(io::IO, bw::BasisWeight)
    print(io, "BasisWeight: \n")
    print(io, "  Basis: ", basis(bw), "\n")
    print(io, "  Property names: ")
    print(io, join(map(propertynames(bw)) do name
        string(name, "::", typeof(getproperty(bw, name)))
    end, ", "), "\n")
    print(io, "  Support nodes: ", supportnodes(bw))
end

"""
    BasisWeightArray

Structure-of-arrays storage for multiple [`BasisWeight`](@ref)s.
Use [`generate_basis_weights`](@ref) to construct a `BasisWeightArray`.
"""
struct BasisWeightArray{B, Prop <: NamedTuple, Indices, ElType <: BasisWeight{B}, N} <: AbstractArray{ElType, N}
    basis::B
    prop::Prop
    indices::Indices
end

function BasisWeightArray(basis::B, prop::Prop, indices::Indices) where {B, Prop <: NamedTuple, N, Indices <: AbstractArray{<: Any, N}}
    ElType = Base._return_type(_getindex, Tuple{B, Prop, Indices, Vararg{Int, N}})
    BasisWeightArray{B, Prop, Indices, ElType, N}(basis, prop, indices)
end

# AbstractMesh
function _generate_basis_weights(::Type{T}, basis, mesh::AbstractMesh{dim}, dims::Dims{N}; kwargs...) where {T, dim, N}
    prop = map(create_property(Vec{dim, T}, basis; kwargs...)) do prop
        fill(zero(eltype(prop)), size(prop)..., dims...)
    end
    indices = map(p->initial_supportnodes(basis, mesh), CartesianIndices(dims))
    BasisWeightArray(basis, prop, indices)
end

_todims(x::Tuple{Vararg{Int}}) = x
_todims(x::Vararg{Int}) = x

"""
    generate_basis_weights([T,] ::Basis, mesh, dims...)
    generate_basis_weights([T,] ::UnstructuredMesh, dims...)

Generate an array of [`BasisWeight`](@ref)s for `basis` on `mesh`.
For unstructured meshes, the mesh cell shape is used as the basis.
"""
function generate_basis_weights end

# CartesianMesh
generate_basis_weights(::Type{T}, basis::Basis, mesh::CartesianMesh, dims...; kwargs...) where {T} = _generate_basis_weights(T, basis, mesh, _todims(dims...); kwargs...)
generate_basis_weights(basis::Basis, mesh::CartesianMesh, dims...; kwargs...) = _generate_basis_weights(Float64, basis, mesh, _todims(dims...); kwargs...)

# UnstructuredMesh
generate_basis_weights(::Type{T}, mesh::UnstructuredMesh, dims...; kwargs...) where {T} = _generate_basis_weights(T, cellshape(mesh), mesh, _todims(dims...); kwargs...)
generate_basis_weights(mesh::UnstructuredMesh, dims...; kwargs...) = _generate_basis_weights(Float64, cellshape(mesh), mesh, _todims(dims...); kwargs...)

Base.size(x::BasisWeightArray) = size(getfield(x, :indices))

Base.propertynames(x::BasisWeightArray) = propertynames(getfield(x, :prop))
@inline function Base.getproperty(x::BasisWeightArray, name::Symbol)
    getproperty(getfield(x, :prop), name)
end

@inline basis(x::BasisWeightArray) = getfield(x, :basis)

@inline function Base.getindex(x::BasisWeightArray{<: Any, <: Any, <: Any, <: Any, N}, I::Vararg{Integer, N}) where {N}
    @boundscheck checkbounds(x, I...)
    @inbounds _getindex(getfield(x, :basis), getfield(x, :prop), getfield(x, :indices), I...)
end
@generated function _getindex(basis, prop::NamedTuple{names}, indices::AbstractArray{<: Any, N}, I::Vararg{Integer, N}) where {names, N}
    exps = [:(viewcol(prop.$name, I...)) for name in names]
    quote
        @_inline_meta
        @_propagate_inbounds_meta
        BasisWeight(basis, NamedTuple{names}(tuple($(exps...))), view(indices, map(:, I, I)...))
    end
end

@inline function viewcol(A::AbstractArray, I::Vararg{Integer, N}) where {N}
    colons = nfill(:, Val(ndims(A)-N))
    @boundscheck checkbounds(A, colons..., I...)
    @inbounds view(A, colons..., I...)
end

function _show_basis_weight_array(io::IO, weights::BasisWeightArray)
    bw = first(weights)
    print(io, Base.dims2string(size(weights)), " ", ndims(weights)==1 ? "BasisWeightVector" : "BasisWeightArray", ": \n")
    print(io, "  Basis: ", basis(weights), "\n")
    print(io, "  Property names: ", join(propertynames(bw), ", "))
end

Base.show(io::IO, ::MIME"text/plain", weights::BasisWeightArray) = _show_basis_weight_array(io, weights)
Base.show(io::IO, weights::BasisWeightArray) = _show_basis_weight_array(io, weights)

###########
# update! #
###########

@inline function alltrue(A::AbstractArray{Bool}, indices::CartesianIndices)
    @debug checkbounds(A, indices)
    @inbounds for i in indices
        A[i] || return false
    end
    true
end
@inline function alltrue(A::Trues, indices::CartesianIndices)
    @debug checkbounds(A, indices)
    true
end

@inline has_full_support(bw::BasisWeight, indices) = size(values(bw, 1)) == size(indices)
@inline has_full_support(bw::BasisWeight, indices, ::Trues) = has_full_support(bw, indices)
@inline has_full_support(bw::BasisWeight, indices, filter::AbstractArray{Bool}) = has_full_support(bw, indices) && alltrue(filter, indices)

@inline function update!(bw::BasisWeight, pt, mesh::AbstractMesh)
    b = basis(bw)
    supportnodes_storage(bw)[] = supportnodes(b, pt, mesh)
    update_property!(bw, b, pt, mesh)
    bw
end
@inline function update!(bw::BasisWeight, pt, mesh::AbstractMesh, filter::AbstractArray{Bool})
    @assert size(mesh) == size(filter)
    b = basis(bw)
    supportnodes_storage(bw)[] = supportnodes(b, pt, mesh)
    update_property!(bw, b, pt, mesh, filter)
    bw
end
@inline update!(bw::BasisWeight, pt, mesh::AbstractMesh, ::Trues) = update!(bw, pt, mesh)
@inline function update_property!(bw::BasisWeight, basis, pt, mesh::AbstractMesh, filter)
    @assert filter isa Trues
    update_property!(bw, basis, pt, mesh)
end

# accelerations

@kernel function gpukernel_update_weight(weights, @Const(particles), @Const(mesh), @Const(filter))
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

where [`LaxyRow`](https://juliaarrays.github.io/StructArrays.jl/stable/#Lazy-row-iteration) is provided in [StructArrays.jl](https://github.com/JuliaArrays/StructArrays.jl).
"""
function update!(weights::AbstractArray{<: BasisWeight}, particles::StructArray, mesh::AbstractMesh, filter::AbstractArray=Trues(size(mesh)))
    @assert length(weights) ≥ length(particles)

    # check backend
    backend = get_backend(weights)
    @assert get_backend(weights) == get_backend(particles) == get_backend(mesh) == backend
    @assert filter isa Trues || get_backend(filter) == backend

    if backend isa CPU
        @threaded for p in 1:length(particles)
            @inbounds update!(weights[p], LazyRow(particles, p), mesh, filter)
        end
    else
        kernel = gpukernel_update_weight(backend)
        kernel(weights, particles, mesh, filter; ndrange=length(particles))
        synchronize(backend)
    end
    weights
end
