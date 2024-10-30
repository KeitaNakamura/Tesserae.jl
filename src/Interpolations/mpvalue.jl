abstract type Interpolation end
abstract type Kernel <: Interpolation end

struct Order{n}
    Order{n}() where {n} = new{n::Int}()
end
Order(n::Int) = Order{n}()

struct Degree{n}
    Degree{n}() where {n} = new{n::Int}()
end
Degree(n::Int) = Degree{n}()
const Linear    = Degree{1}
const Quadratic = Degree{2}
const Cubic     = Degree{3}
const Quartic   = Degree{4}

#=
To create a new interpolation, following methods need to be implemented.
* Tesserae.create_property(::Type{T}, it::Interpolation, mesh; kwargs...) -> NamedTuple
* Tesserae.initial_neighboringnodes(it::Interpolation, mesh)
* Tesserae.update!(mp::MPValue, it::Interpolation, pt, mesh)
=#

initial_neighboringnodes(::Interpolation, ::CartesianMesh{dim}) where {dim} = EmptyCartesianIndices(Val(dim))

@generated function create_property(::Type{T}, it::Interpolation, mesh::CartesianMesh{dim}; derivative::Order{k}=Order(1)) where {dim, T, k}
    quote
        dims = nfill(gridspan(it), Val(dim))
        names = @ntuple $(k+1) i -> create_name(Order(i-1))
        vals = @ntuple $(k+1) i -> fill(zero(create_elval(Vec{dim, T}, Order(i-1))), dims)
        NamedTuple{names}(vals)
    end
end

create_elval(::Type{Vec{dim, T}}, ::Order{0}) where {dim, T} = zero(T)
create_elval(::Type{Vec{dim, T}}, ::Order{1}) where {dim, T} = zero(Vec{dim, T})
create_elval(::Type{Vec{dim, T}}, ::Order{k}) where {dim, T, k} = zero(Tensor{Tuple{@Symmetry{ntuple(i->dim, k)...}}, T})
create_name(::Order{0}) = :w
create_name(::Order{1}) = :∇w
create_name(::Order{2}) = :∇²w
create_name(::Order{3}) = :∇³w
create_name(::Order{4}) = :∇⁴w
create_name(::Order{5}) = :∇⁵w
create_name(::Order{6}) = :∇⁶w
create_name(::Order{7}) = :∇⁷w
create_name(::Order{8}) = :∇⁸w
create_name(::Order{9}) = :∇⁹w

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
struct MPValue{It, Prop <: NamedTuple, Indices <: AbstractArray{<: Any, 0}}
    it::It
    prop::Prop
    indices::Indices
end

function MPValue(::Type{T}, it::Interpolation, mesh::AbstractMesh; kwargs...) where {T}
    prop = create_property(T, it, mesh; kwargs...)
    indices = initial_neighboringnodes(it, mesh)
    MPValue(it, prop, fill(indices))
end
MPValue(it::Interpolation, mesh::AbstractMesh; kwargs...) = MPValue(Float64, it, mesh; kwargs...)

Base.propertynames(mp::MPValue) = propertynames(getfield(mp, :prop))
@inline function Base.getproperty(mp::MPValue, name::Symbol)
    getproperty(getfield(mp, :prop), name)
end
@inline function Base.values(mp::MPValue, i::Int)
    getfield(mp, :prop)[i]
end

@inline function check_mpvalue_prop(mp::MPValue)
    k = length(propertynames(mp)) - 1
    _check_mpvalue_prop(mp, Val(k))
end
@generated function _check_mpvalue_prop(mp::MPValue, ::Val{k}) where {k}
    quote
        @_inline_meta
        @assert @nall $(k+1) i -> create_name(Order(i-1)) === propertynames(mp)[i]
    end
end

@inline interpolation(mp::MPValue) = getfield(mp, :it)

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

@inline neighboringnodes_storage(mp::MPValue) = getfield(mp, :indices)

@inline function derivative_order(mp::MPValue)
    check_mpvalue_prop(mp)
    k = length(propertynames(mp)) - 1
    Order(k)
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
    print(io, "  Interpolation: ", interpolation(mp), "\n")
    print(io, "  Property names: ")
    print(io, join(map(propertynames(mp)) do name
        string(name, "::", typeof(getproperty(mp, name)))
    end, ", "), "\n")
    print(io, "  Neighboring nodes: ", neighboringnodes(mp))
end

struct MPValueVector{It, Prop <: NamedTuple, Indices, ElType <: MPValue{It}} <: AbstractVector{ElType}
    it::It
    prop::Prop
    indices::Indices
end

function generate_mpvalues(::Type{T}, it::Interpolation, mesh::AbstractMesh, n::Int; kwargs...) where {T}
    prop = map(create_property(T, it, mesh; kwargs...)) do prop
        fill(zero(eltype(prop)), size(prop)..., n)
    end
    indices = map(p->initial_neighboringnodes(it, mesh), 1:n)
    It = typeof(it)
    Prop = typeof(prop)
    Indices = typeof(indices)
    ElType = Base._return_type(_getindex, Tuple{It, Prop, Indices, Int})
    MPValueVector{It, Prop, Indices, ElType}(it, prop, indices)
end
generate_mpvalues(it::Interpolation, mesh::AbstractMesh, n::Int; kwargs...) = generate_mpvalues(Float64, it, mesh, n; kwargs...)

Base.IndexStyle(::Type{<: MPValueVector}) = IndexLinear()
Base.size(x::MPValueVector) = size(getfield(x, :indices))

Base.propertynames(x::MPValueVector) = propertynames(getfield(x, :prop))
@inline function Base.getproperty(x::MPValueVector, name::Symbol)
    getproperty(getfield(x, :prop), name)
end

@inline interpolation(x::MPValueVector) = getfield(x, :it)

@inline function Base.getindex(x::MPValueVector, i::Integer)
    @boundscheck checkbounds(x, i)
    @inbounds _getindex(getfield(x, :it), getfield(x, :prop), getfield(x, :indices), i)
end
@generated function _getindex(it::Interpolation, prop::NamedTuple{names}, indices, i::Integer) where {names}
    exps = [:(viewcol(prop.$name, i)) for name in names]
    quote
        @_inline_meta
        @_propagate_inbounds_meta
        MPValue(it, NamedTuple{names}(tuple($(exps...))), view(indices, i))
    end
end

@inline function viewcol(A::AbstractArray, i::Integer)
    @boundscheck checkbounds(axes(A, ndims(A)), i)
    colons = nfill(:, Val(ndims(A)-1))
    @inbounds view(A, colons..., i)
end

function Base.show(io::IO, mime::MIME"text/plain", mpvalues::MPValueVector)
    mp = first(mpvalues)
    print(io, length(mpvalues), "-element MPValueVector: \n")
    print(io, "  Interpolation: ", interpolation(mpvalues), "\n")
    print(io, "  Property names: ", join(propertynames(mp), ", "))
end

function Base.show(io::IO, mpvalues::MPValueVector)
    mp = first(mpvalues)
    print(io, length(mpvalues), "-element MPValueVector: \n")
    print(io, "  Interpolation: ", interpolation(mpvalues), "\n")
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
    it = interpolation(mp)
    neighboringnodes_storage(mp)[] = neighboringnodes(it, pt, mesh)
    update_property!(mp, it, pt, mesh)
    mp
end
function update!(mp::MPValue, pt, mesh::AbstractMesh, filter::AbstractArray{Bool})
    @debug @assert size(mesh) == size(filter)
    it = interpolation(mp)
    neighboringnodes_storage(mp)[] = neighboringnodes(it, pt, mesh)
    update_property!(mp, it, pt, mesh, filter)
    mp
end
