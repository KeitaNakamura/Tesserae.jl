abstract type Interpolation end
abstract type Kernel <: Interpolation end

#=
To create a new interpolation, following methods need to be implemented.
* Tesserae.create_property(::Type{Vec{dim, T}}, interp::Interpolation; kwargs...) -> NamedTuple
* Tesserae.initial_neighboringnodes(interp::Interpolation, mesh)
* Tesserae.update!(mp::MPValue, interp::Interpolation, pt, mesh)
=#

initial_neighboringnodes(::Interpolation, ::CartesianMesh{dim}) where {dim} = EmptyCartesianIndices(Val(dim))
initial_neighboringnodes(shape::Shape, mesh::UnstructuredMesh) = zero(SVector{nlocalnodes(shape), Int})

propsize(interp::Interpolation, ::Val{dim}) where{dim} = nfill(gridspan(interp), Val(dim))
propsize(shape::Shape, ::Val)  = (nlocalnodes(shape),)
@generated function create_property(::Type{Vec{dim, T}}, interp; derivative::Order{k}=Order(1), name::Val=Val(:w)) where {dim, T, k}
    quote
        dims = propsize(interp, Val(dim))
        names = @ntuple $(k+1) i -> create_name(Order(i-1), name)
        vals = @ntuple $(k+1) i -> fill(zero(create_elval(Vec{dim, T}, Order(i-1))), dims)
        NamedTuple{names}(vals)
    end
end

create_elval(::Type{Vec{dim, T}}, ::Order{0}) where {dim, T} = zero(T)
create_elval(::Type{Vec{dim, T}}, ::Order{1}) where {dim, T} = zero(Vec{dim, T})
create_elval(::Type{Vec{dim, T}}, ::Order{k}) where {dim, T, k} = zero(Tensor{Tuple{@Symmetry{ntuple(i->dim, k)...}}, T})
create_name(::Order{0}, ::Val{name}) where {name} = name
create_name(::Order{1}, ::Val{name}) where {name} = Symbol(:∇, name)
create_name(::Order{2}, ::Val{name}) where {name} = Symbol(:∇², name)
create_name(::Order{3}, ::Val{name}) where {name} = Symbol(:∇³, name)
create_name(::Order{4}, ::Val{name}) where {name} = Symbol(:∇⁴, name)
create_name(::Order{5}, ::Val{name}) where {name} = Symbol(:∇⁵, name)
create_name(::Order{6}, ::Val{name}) where {name} = Symbol(:∇⁶, name)
create_name(::Order{7}, ::Val{name}) where {name} = Symbol(:∇⁷, name)
create_name(::Order{8}, ::Val{name}) where {name} = Symbol(:∇⁸, name)
create_name(::Order{9}, ::Val{name}) where {name} = Symbol(:∇⁹, name)

@generated function prod_each_dimension(::Order{0}, vals::Vararg{Tuple, dim}) where {dim}
    quote
        @_inline_meta
        tuple_otimes(@ntuple $dim d -> vals[d][1])
    end
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
    MPValue([T,] interpolation, mesh)

`MPValue` stores properties for interpolation, such as the basis function values and its derivatives.

```jldoctest
julia> mesh = CartesianMesh(1.0, (0,5), (0,5));

julia> xₚ = Vec(2.2, 3.4); # particle position

julia> mp = MPValue(BSpline(Quadratic()), mesh);

julia> update!(mp, xₚ, mesh) # update `mp` at position `xₚ` in `mesh`
MPValue:
  Interpolation: BSpline(Quadratic())
  Property names: w::Matrix{Float64}, ∇w::Matrix{Vec{2, Float64}}
  Neighboring nodes: CartesianIndices((2:4, 3:5))

julia> sum(mp.w) ≈ 1 # partition of unity
true

julia> nodeindices = neighboringnodes(mp) # grid indices within a particles' local domain
CartesianIndices((2:4, 3:5))

julia> sum(eachindex(nodeindices)) do ip # linear field reproduction
           i = nodeindices[ip]
           mp.w[ip] * mesh[i]
       end ≈ xₚ
true
```
"""
struct MPValue{Interp, Prop <: NamedTuple, Indices <: AbstractArray{<: Any}}
    interp::Interp
    prop::Prop
    indices::Indices
end

# AbstractMesh
function _MPValue(::Type{T}, interp, mesh::AbstractMesh{dim}; kwargs...) where {T, dim}
    prop = create_property(Vec{dim, T}, interp; kwargs...)
    indices = initial_neighboringnodes(interp, mesh)
    MPValue(interp, prop, fill(indices))
end

# CartesianMesh
MPValue(::Type{T}, interp::Interpolation, mesh::CartesianMesh; kwargs...) where {T} = _MPValue(T, interp, mesh; kwargs...)
MPValue(interp::Interpolation, mesh::CartesianMesh; kwargs...) = _MPValue(Float64, interp, mesh; kwargs...)

# UnstructuredMesh
MPValue(::Type{T}, mesh::UnstructuredMesh; kwargs...) where {T} = _MPValue(T, cellshape(mesh), mesh; kwargs...)
MPValue(mesh::UnstructuredMesh; kwargs...) = MPValue(Float64, mesh; kwargs...)

Base.propertynames(mp::MPValue) = propertynames(getfield(mp, :prop))
@inline function Base.getproperty(mp::MPValue, name::Symbol)
    getproperty(getfield(mp, :prop), name)
end
@inline function Base.values(mp::MPValue, i::Int)
    getfield(mp, :prop)[i]
end

@inline interpolation(mp::MPValue) = getfield(mp, :interp)::Interpolation
@inline cellshape(mp::MPValue) = getfield(mp, :interp)::Shape

@inline neighboringnodes(mp::MPValue) = getfield(mp, :indices)[]
@inline function neighboringnodes(mp::MPValue, grid::Grid)
    neighboringnodes(mp, get_mesh(grid))
end
@inline function neighboringnodes(mp::MPValue, mesh::CartesianMesh)
    inds = neighboringnodes(mp)
    @boundscheck checkbounds(mesh, inds)
    inds
end
# SpGrid always use CartesianMesh
@inline function neighboringnodes(mp::MPValue, grid::SpGrid)
    inds = neighboringnodes(mp)
    spinds = get_spinds(grid)
    @boundscheck checkbounds(spinds, inds)
    @inbounds neighbors = view(spinds, inds)
    @debug @assert all(isactive, neighbors)
    neighbors
end

@inline function neighboringnodes(mp::MPValue, mesh::UnstructuredMesh)
    inds = neighboringnodes(mp)
    @boundscheck checkbounds(mesh, inds)
    inds
end

@inline neighboringnodes_storage(mp::MPValue) = getfield(mp, :indices)

@inline function derivative_order(mp::MPValue)
    @debug check_mpvalue_prop(mp)
    k = length(propertynames(mp)) - 1
    Order(k)
end
@inline function check_mpvalue_prop(mp::MPValue)
    k = length(propertynames(mp)) - 1
    _check_mpvalue_prop(mp, Val(k))
end
@generated function _check_mpvalue_prop(mp::MPValue, ::Val{k}) where {k}
    quote
        @_inline_meta
        @assert @nall $(k+1) i -> create_name(Order(i-1), Val(propertynames(mp)[1])) === propertynames(mp)[i]
    end
end

@generated function set_values!(mp::MPValue, ip, vals::Tuple{Vararg{Any, N}}) where {N}
    quote
        @_inline_meta
        @_propagate_inbounds_meta
        @nexprs $N i -> values(mp, i)[ip] = vals[i]
    end
end
@generated function set_values!(mp::MPValue, vals::Tuple{Vararg{Any, N}}) where {N}
    quote
        @_inline_meta
        @nexprs $N i -> copyto!(values(mp, i), vals[i])
    end
end

function Base.show(io::IO, mp::MPValue)
    print(io, "MPValue: \n")
    print(io, "  Interpolation: ", getfield(mp, :interp), "\n")
    print(io, "  Property names: ")
    print(io, join(map(propertynames(mp)) do name
        string(name, "::", typeof(getproperty(mp, name)))
    end, ", "), "\n")
    print(io, "  Neighboring nodes: ", neighboringnodes(mp))
end

struct MPValueArray{Interp, Prop <: NamedTuple, Indices, ElType <: MPValue{Interp}, N} <: AbstractArray{ElType, N}
    interp::Interp
    prop::Prop
    indices::Indices
end

# AbstractMesh
function _generate_mpvalues(::Type{T}, interp, mesh::AbstractMesh{dim}, dims::Dims{N}; kwargs...) where {T, dim, N}
    prop = map(create_property(Vec{dim, T}, interp; kwargs...)) do prop
        fill(zero(eltype(prop)), size(prop)..., dims...)
    end
    indices = map(p->initial_neighboringnodes(interp, mesh), CartesianIndices(dims))
    Interp = typeof(interp)
    Prop = typeof(prop)
    Indices = typeof(indices)
    ElType = Base._return_type(_getindex, Tuple{Interp, Prop, Indices, Int})
    MPValueArray{Interp, Prop, Indices, ElType, N}(interp, prop, indices)
end

_todims(x::Tuple{Vararg{Int}}) = x
_todims(x::Vararg{Int}) = x
# CartesianMesh
generate_mpvalues(::Type{T}, interp::Interpolation, mesh::CartesianMesh, dims...; kwargs...) where {T} = _generate_mpvalues(T, interp, mesh, _todims(dims...); kwargs...)
generate_mpvalues(interp::Interpolation, mesh::CartesianMesh, dims...; kwargs...) = _generate_mpvalues(Float64, interp, mesh, _todims(dims...); kwargs...)

# UnstructuredMesh
generate_mpvalues(::Type{T}, mesh::UnstructuredMesh, dims...; kwargs...) where {T} = _generate_mpvalues(T, interp, mesh, _todims(dims...); kwargs...)
generate_mpvalues(mesh::UnstructuredMesh, dims...; kwargs...) = _generate_mpvalues(Float64, cellshape(mesh), mesh, _todims(dims...); kwargs...)

Base.size(x::MPValueArray) = size(getfield(x, :indices))

Base.propertynames(x::MPValueArray) = propertynames(getfield(x, :prop))
@inline function Base.getproperty(x::MPValueArray, name::Symbol)
    getproperty(getfield(x, :prop), name)
end

@inline interpolation(x::MPValueArray) = getfield(x, :interp)::Interpolation
@inline cellshape(x::MPValueArray) = getfield(x, :interp)::Shape

@inline function Base.getindex(x::MPValueArray{<: Any, <: Any, <: Any, <: Any, N}, I::Vararg{Integer, N}) where {N}
    @boundscheck checkbounds(x, I...)
    @inbounds _getindex(getfield(x, :interp), getfield(x, :prop), getfield(x, :indices), I...)
end
@generated function _getindex(interp, prop::NamedTuple{names}, indices::AbstractArray{<: Any, N}, I::Vararg{Integer, N}) where {names, N}
    exps = [:(viewcol(prop.$name, I...)) for name in names]
    quote
        @_inline_meta
        @_propagate_inbounds_meta
        MPValue(interp, NamedTuple{names}(tuple($(exps...))), view(indices, map(:, I, I)...))
    end
end

@inline function viewcol(A::AbstractArray, I::Vararg{Integer, N}) where {N}
    colons = nfill(:, Val(ndims(A)-N))
    @boundscheck checkbounds(A, colons..., I...)
    @inbounds view(A, colons..., I...)
end

function Base.show(io::IO, mime::MIME"text/plain", mpvalues::MPValueArray)
    mp = first(mpvalues)
    print(io, Base.dims2string(size(mpvalues)), " ", ndims(mpvalues)==1 ? "MPValueVector" : "MPValueArray", ": \n")
    print(io, "  Interpolation: ", getfield(mpvalues, :interp), "\n")
    print(io, "  Property names: ", join(propertynames(mp), ", "))
end

function Base.show(io::IO, mpvalues::MPValueArray)
    mp = first(mpvalues)
    print(io, Base.dims2string(size(mpvalues)), " ", ndims(mpvalues)==1 ? "MPValueVector" : "MPValueArray", ": \n")
    print(io, "  Interpolation: ", getfield(mpvalues, :interp), "\n")
    print(io, "  Property names: ", join(propertynames(mp), ", "))
end

###########
# update! #
###########

@inline function alltrue(A::AbstractArray{Bool}, indices::CartesianIndices)
    @debug checkbounds(A, indices)
    @inbounds @simd for i in indices
        A[i] || return false
    end
    true
end
@inline function alltrue(A::Trues, indices::CartesianIndices)
    @debug checkbounds(A, indices)
    true
end

function update!(mp::MPValue, pt, mesh::AbstractMesh)
    interp = interpolation(mp)
    neighboringnodes_storage(mp)[] = neighboringnodes(interp, pt, mesh)
    update_property!(mp, interp, pt, mesh)
    mp
end
function update!(mp::MPValue, pt, mesh::AbstractMesh, filter::AbstractArray{Bool})
    @assert size(mesh) == size(filter)
    interp = interpolation(mp)
    neighboringnodes_storage(mp)[] = neighboringnodes(interp, pt, mesh)
    update_property!(mp, interp, pt, mesh, filter)
    mp
end
function update_property!(mp::MPValue, interp, pt, mesh::AbstractMesh, filter)
    @assert filter isa Trues
    update_property!(mp, interp, pt, mesh)
end

# accelerations

@kernel function gpukernel_update_mpvalue(mpvalues, @Const(particles), @Const(mesh), @Const(filter))
    p = @index(Global)
    update!(mpvalues[p], LazyRow(particles, p), mesh, filter)
end

function update!(mpvalues::MPValueArray, particles::StructArray, mesh::AbstractMesh, filter::AbstractArray=Trues(size(mesh)))
    @assert length(mpvalues) == length(particles)

    # check backend
    backend = get_backend(mpvalues)
    @assert get_backend(mpvalues) == get_backend(particles) == get_backend(mesh) == backend
    @assert filter isa Trues || get_backend(filter) == backend

    if backend isa CPU
        @threaded for p in 1:length(particles)
            @inbounds update!(mpvalues[p], LazyRow(particles, p), mesh, filter)
        end
    else
        kernel = gpukernel_update_mpvalue(backend, 256)
        kernel(mpvalues, particles, mesh, filter; ndrange=length(particles))
        synchronize(backend)
    end
    mpvalues
end
