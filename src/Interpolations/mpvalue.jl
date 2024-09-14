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

function create_property(::Type{T}, it::Interpolation, mesh::CartesianMesh{dim}; derivative::Order=Order(1)) where {dim, T}
    dims = nfill(gridspan(it), Val(dim))
    derivative isa Order{0} && return (; w=zeros(T, dims))
    derivative isa Order{1} && return (; w=fill(zero(T), dims), ∇w=fill(zero(Vec{dim, T}), dims))
    derivative isa Order{2} && return (; w=fill(zero(T), dims), ∇w=fill(zero(Vec{dim, T}), dims), ∇∇w=fill(zero(SymmetricSecondOrderTensor{dim, T}), dims))
    derivative isa Order{3} && return (; w=fill(zero(T), dims), ∇w=fill(zero(Vec{dim, T}), dims), ∇∇w=fill(zero(SymmetricSecondOrderTensor{dim, T}), dims), ∇∇∇w=fill(zero(Tensor{Tuple{@Symmetry{dim,dim,dim}}, T}), dims))
    error("wrong derivative type, got $derivative")
end

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
    hasproperty(mp, :∇∇∇w) && return Order(3)
    hasproperty(mp, :∇∇w)  && return Order(2)
    hasproperty(mp, :∇w)   && return Order(1)
    hasproperty(mp, :w)    && return Order(0)
    error("unreachable")
end
@inline @propagate_inbounds value(f, x, args...) = f(x, args...)
@inline @propagate_inbounds value(::Order{0}, f, x, args...) = (value(f, x, args...),)
@inline @propagate_inbounds value(::Order{1}, f, x, args...) = reverse(gradient(x -> (@_inline_meta; @_propagate_inbounds_meta; value(f, x, args...)), x, :all))
@inline @propagate_inbounds value(::Order{2}, f, x, args...) = reverse(hessian(x -> (@_inline_meta; @_propagate_inbounds_meta; value(f, x, args...)), x, :all))
@inline @propagate_inbounds function value(::Order{3}, f, x, args...)
    @inline function ∇∇f(x)
        @_propagate_inbounds_meta
        hessian(x -> (@_inline_meta; @_propagate_inbounds_meta; value(f, x, args...)), x)
    end
    (value(Order(2), f, x, args...)..., gradient(∇∇f, x))
end

@inline @propagate_inbounds set_kernel_values!(mp::MPValue, ip, (w,)::Tuple{Any}) = (mp.w[ip]=w;)
@inline @propagate_inbounds set_kernel_values!(mp::MPValue, ip, (w,∇w)::Tuple{Any,Any}) = (mp.w[ip]=w; mp.∇w[ip]=∇w;)
@inline @propagate_inbounds set_kernel_values!(mp::MPValue, ip, (w,∇w,∇∇w)::Tuple{Any,Any,Any}) = (mp.w[ip]=w; mp.∇w[ip]=∇w; mp.∇∇w[ip]=∇∇w;)
@inline @propagate_inbounds set_kernel_values!(mp::MPValue, ip, (w,∇w,∇∇w,∇∇∇w)::Tuple{Any,Any,Any,Any}) = (mp.w[ip]=w; mp.∇w[ip]=∇w; mp.∇∇w[ip]=∇∇w; mp.∇∇∇w[ip]=∇∇∇w;)
@inline set_kernel_values!(mp::MPValue, (w,)::Tuple{Any}) = copyto!(mp.w, w)
@inline set_kernel_values!(mp::MPValue, (w,∇w)::Tuple{Any,Any}) = (copyto!(mp.w,w); copyto!(mp.∇w,∇w);)
@inline set_kernel_values!(mp::MPValue, (w,∇w,∇∇w)::Tuple{Any,Any,Any}) = (copyto!(mp.w,w); copyto!(mp.∇w,∇w); copyto!(mp.∇∇w,∇∇w);)
@inline set_kernel_values!(mp::MPValue, (w,∇w,∇∇w,∇∇∇w)::Tuple{Any,Any,Any,Any}) = (copyto!(mp.w,w); copyto!(mp.∇w,∇w); copyto!(mp.∇∇w,∇∇w); copyto!(mp.∇∇∇w,∇∇∇w);)

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
