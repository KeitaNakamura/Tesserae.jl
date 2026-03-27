abstract type Interpolation end
abstract type Kernel <: Interpolation end

#=
To create a new interpolation, following methods need to be implemented.
* Tesserae.create_property(::Type{Vec{dim, T}}, interp::Interpolation; kwargs...) -> NamedTuple
* Tesserae.initial_neighboringnodes(interp::Interpolation, mesh)
* Tesserae.update!(iw::InterpolationWeight, interp::Interpolation, pt, mesh)
=#

initial_neighboringnodes(::Interpolation, ::CartesianMesh{dim}) where {dim} = EmptyCartesianIndices(Val(dim))
initial_neighboringnodes(shape::Shape, mesh::UnstructuredMesh) = zero(SVector{nlocalnodes(shape), Int})

propsize(interp::Interpolation, ::Val{dim}) where{dim} = nfill(kernel_support(interp), Val(dim))
propsize(shape::Shape, ::Val)  = (nlocalnodes(shape),)
function create_property(::Type{Vec{dim, T}}, interp; derivative::Order{k}=Order(1), name=nothing) where {dim, T, k}
    map(Array, create_property(MArray, Vec{dim, T}, interp; derivative, name))
end
@generated function create_property(::Type{MArray}, ::Type{Vec{dim, T}}, interp; derivative::Order{k}=Order(1), name=nothing) where {dim, T, k}
    quote
        arrdims = propsize(interp, Val(dim))
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
        v = $(Expr(:tuple, v[Tensorial.tensorindices_tuple(TT)]...))
        map($TT, map(tuple_otimes, v)...)
    end
end
@inline tuple_otimes(x::Tuple) = SArray(⊗(map(Vec, x)...))

"""
    InterpolationWeight([T,] interpolation, mesh)

`InterpolationWeight` stores interpolation data, such as basis function values and their spatial derivatives.

```jldoctest
julia> mesh = CartesianMesh(1.0, (0,5), (0,5));

julia> xₚ = Vec(2.2, 3.4); # particle position

julia> iw = InterpolationWeight(BSpline(Quadratic()), mesh);

julia> update!(iw, xₚ, mesh) # update `iw` at position `xₚ` in `mesh`
InterpolationWeight:
  Interpolation: BSpline(Quadratic())
  Property names: w::Matrix{Float64}, ∇w::Matrix{Vec{2, Float64}}
  Neighboring nodes: CartesianIndices((2:4, 3:5))

julia> sum(iw.w) ≈ 1 # partition of unity
true

julia> nodeindices = neighboringnodes(iw) # grid indices within a particles' local domain
CartesianIndices((2:4, 3:5))

julia> sum(eachindex(nodeindices)) do ip # linear field reproduction
           i = nodeindices[ip]
           iw.w[ip] * mesh[i]
       end ≈ xₚ
true
```
"""
struct InterpolationWeight{Interp, Prop <: NamedTuple, Indices <: AbstractArray{<: Any}}
    interp::Interp
    prop::Prop
    indices::Indices
end

# AbstractMesh
function _InterpolationWeight(::Type{T}, interp, mesh::AbstractMesh{dim}; kwargs...) where {T, dim}
    prop = create_property(Vec{dim, T}, interp; kwargs...)
    indices = initial_neighboringnodes(interp, mesh)
    InterpolationWeight(interp, prop, fill(indices))
end

# CartesianMesh
InterpolationWeight(::Type{T}, interp::Interpolation, mesh::CartesianMesh; kwargs...) where {T} = _InterpolationWeight(T, interp, mesh; kwargs...)
InterpolationWeight(interp::Interpolation, mesh::CartesianMesh; kwargs...) = _InterpolationWeight(Float64, interp, mesh; kwargs...)

# UnstructuredMesh
InterpolationWeight(::Type{T}, mesh::UnstructuredMesh; kwargs...) where {T} = _InterpolationWeight(T, cellshape(mesh), mesh; kwargs...)
InterpolationWeight(mesh::UnstructuredMesh; kwargs...) = InterpolationWeight(Float64, mesh; kwargs...)

Base.propertynames(iw::InterpolationWeight) = propertynames(getfield(iw, :prop))
@inline function Base.getproperty(iw::InterpolationWeight, name::Symbol)
    getproperty(getfield(iw, :prop), name)
end
@inline function Base.values(iw::InterpolationWeight, i::Int)
    getfield(iw, :prop)[i]
end

@inline scalartype(iw::InterpolationWeight) = eltype(values(iw, 1))

@inline interpolation(iw::InterpolationWeight) = getfield(iw, :interp)::Interpolation
@inline cellshape(iw::InterpolationWeight) = getfield(iw, :interp)::Shape

@inline neighboringnodes(iw::InterpolationWeight) = getfield(iw, :indices)[]
@inline function neighboringnodes(iw::InterpolationWeight, grid::Grid)
    neighboringnodes(iw, get_mesh(grid))
end
@inline function neighboringnodes(iw::InterpolationWeight, mesh::CartesianMesh)
    inds = neighboringnodes(iw)
    @boundscheck checkbounds(mesh, inds)
    inds
end
# SpGrid always use CartesianMesh
@inline function neighboringnodes(iw::InterpolationWeight, grid::SpGrid)
    inds = neighboringnodes(iw)
    spinds = get_spinds(grid)
    @boundscheck checkbounds(spinds, inds)
    @inbounds neighbors = view(spinds, inds)
    @debug @assert all(isactive, neighbors)
    neighbors
end

@inline function neighboringnodes(iw::InterpolationWeight, mesh::UnstructuredMesh)
    inds = neighboringnodes(iw)
    @boundscheck checkbounds(mesh, inds)
    inds
end

@inline neighboringnodes_storage(iw::InterpolationWeight) = getfield(iw, :indices)

@inline function derivative_order(iw::InterpolationWeight)
    @debug check_weight_prop(iw)
    k = length(propertynames(iw)) - 1
    Order(k)
end
@inline function check_weight_prop(iw::InterpolationWeight)
    k = length(propertynames(iw)) - 1
    _check_weight_prop(iw, Val(k))
end
@generated function _check_weight_prop(iw::InterpolationWeight, ::Val{k}) where {k}
    quote
        @_inline_meta
        @assert @nall $(k+1) i -> create_name(Order(i-1), Val(propertynames(iw)[1])) === propertynames(iw)[i]
    end
end

@generated function set_values!(iw::InterpolationWeight, ip, vals::Tuple{Vararg{Any, N}}) where {N}
    quote
        @_inline_meta
        @_propagate_inbounds_meta
        @nexprs $N i -> values(iw, i)[ip] = vals[i]
    end
end
@generated function set_values!(iw::InterpolationWeight, vals::Tuple{Vararg{Any, N}}) where {N}
    quote
        @_inline_meta
        @nexprs $N i -> copyto!(values(iw, i), vals[i])
    end
end

function Base.show(io::IO, iw::InterpolationWeight)
    print(io, "InterpolationWeight: \n")
    print(io, "  Interpolation: ", getfield(iw, :interp), "\n")
    print(io, "  Property names: ")
    print(io, join(map(propertynames(iw)) do name
        string(name, "::", typeof(getproperty(iw, name)))
    end, ", "), "\n")
    print(io, "  Neighboring nodes: ", neighboringnodes(iw))
end

struct InterpolationWeightArray{Interp, Prop <: NamedTuple, Indices, ElType <: InterpolationWeight{Interp}, N} <: AbstractArray{ElType, N}
    interp::Interp
    prop::Prop
    indices::Indices
end

# AbstractMesh
function _generate_weights(::Type{T}, interp, mesh::AbstractMesh{dim}, dims::Dims{N}; kwargs...) where {T, dim, N}
    prop = map(create_property(Vec{dim, T}, interp; kwargs...)) do prop
        fill(zero(eltype(prop)), size(prop)..., dims...)
    end
    indices = map(p->initial_neighboringnodes(interp, mesh), CartesianIndices(dims))
    Interp = typeof(interp)
    Prop = typeof(prop)
    Indices = typeof(indices)
    ElType = Base._return_type(_getindex, Tuple{Interp, Prop, Indices, Int})
    InterpolationWeightArray{Interp, Prop, Indices, ElType, N}(interp, prop, indices)
end

_todims(x::Tuple{Vararg{Int}}) = x
_todims(x::Vararg{Int}) = x
# CartesianMesh
generate_interpolation_weights(::Type{T}, interp::Interpolation, mesh::CartesianMesh, dims...; kwargs...) where {T} = _generate_weights(T, interp, mesh, _todims(dims...); kwargs...)
generate_interpolation_weights(interp::Interpolation, mesh::CartesianMesh, dims...; kwargs...) = _generate_weights(Float64, interp, mesh, _todims(dims...); kwargs...)

# UnstructuredMesh
generate_interpolation_weights(::Type{T}, mesh::UnstructuredMesh, dims...; kwargs...) where {T} = _generate_weights(T, interp, mesh, _todims(dims...); kwargs...)
generate_interpolation_weights(mesh::UnstructuredMesh, dims...; kwargs...) = _generate_weights(Float64, cellshape(mesh), mesh, _todims(dims...); kwargs...)

Base.size(x::InterpolationWeightArray) = size(getfield(x, :indices))

Base.propertynames(x::InterpolationWeightArray) = propertynames(getfield(x, :prop))
@inline function Base.getproperty(x::InterpolationWeightArray, name::Symbol)
    getproperty(getfield(x, :prop), name)
end

@inline interpolation(x::InterpolationWeightArray) = getfield(x, :interp)::Interpolation
@inline cellshape(x::InterpolationWeightArray) = getfield(x, :interp)::Shape

@inline function Base.getindex(x::InterpolationWeightArray{<: Any, <: Any, <: Any, <: Any, N}, I::Vararg{Integer, N}) where {N}
    @boundscheck checkbounds(x, I...)
    @inbounds _getindex(getfield(x, :interp), getfield(x, :prop), getfield(x, :indices), I...)
end
@generated function _getindex(interp, prop::NamedTuple{names}, indices::AbstractArray{<: Any, N}, I::Vararg{Integer, N}) where {names, N}
    exps = [:(viewcol(prop.$name, I...)) for name in names]
    quote
        @_inline_meta
        @_propagate_inbounds_meta
        InterpolationWeight(interp, NamedTuple{names}(tuple($(exps...))), view(indices, map(:, I, I)...))
    end
end

@inline function viewcol(A::AbstractArray, I::Vararg{Integer, N}) where {N}
    colons = nfill(:, Val(ndims(A)-N))
    @boundscheck checkbounds(A, colons..., I...)
    @inbounds view(A, colons..., I...)
end

function Base.show(io::IO, mime::MIME"text/plain", weights::InterpolationWeightArray)
    iw = first(weights)
    print(io, Base.dims2string(size(weights)), " ", ndims(weights)==1 ? "InterpolationWeightVector" : "InterpolationWeightArray", ": \n")
    print(io, "  Interpolation: ", getfield(weights, :interp), "\n")
    print(io, "  Property names: ", join(propertynames(iw), ", "))
end

function Base.show(io::IO, weights::InterpolationWeightArray)
    iw = first(weights)
    print(io, Base.dims2string(size(weights)), " ", ndims(weights)==1 ? "InterpolationWeightVector" : "InterpolationWeightArray", ": \n")
    print(io, "  Interpolation: ", getfield(weights, :interp), "\n")
    print(io, "  Property names: ", join(propertynames(iw), ", "))
end

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

@inline function update!(iw::InterpolationWeight, pt, mesh::AbstractMesh)
    interp = interpolation(iw)
    neighboringnodes_storage(iw)[] = neighboringnodes(interp, pt, mesh)
    update_property!(iw, interp, pt, mesh)
    iw
end
@inline function update!(iw::InterpolationWeight, pt, mesh::AbstractMesh, filter::AbstractArray{Bool})
    @assert size(mesh) == size(filter)
    interp = interpolation(iw)
    neighboringnodes_storage(iw)[] = neighboringnodes(interp, pt, mesh)
    update_property!(iw, interp, pt, mesh, filter)
    iw
end
@inline update!(iw::InterpolationWeight, pt, mesh::AbstractMesh, ::Trues) = update!(iw, pt, mesh)
@inline function update_property!(iw::InterpolationWeight, interp, pt, mesh::AbstractMesh, filter)
    @assert filter isa Trues
    update_property!(iw, interp, pt, mesh)
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
function update!(weights::AbstractArray{<: InterpolationWeight}, particles::StructArray, mesh::AbstractMesh, filter::AbstractArray=Trues(size(mesh)))
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
