abstract type Interpolation end
abstract type Kernel <: Interpolation end

Broadcast.broadcastable(interp::Interpolation) = (interp,)

# used for constructing `MPValues`
struct MPValuesInfo{dim, T, V <: NamedTuple, S <: Tuple{Vararg{Dims}}}
    values::V
    sizes::S
end
function MPValuesInfo{dim, T}(values::NamedTuple, sizes::Tuple{Vararg{Dims}}) where {dim, T}
    @assert length(values) == length(sizes)
    MPValuesInfo{dim, T, typeof(values), typeof(sizes)}(values, sizes)
end

"""
    MPValues{dim}(interpolation, length)
    MPValues{dim, T}(interpolation, length)
"""
struct MPValues{dim, T, V <: NamedTuple, VI <: AbstractVector{CartesianIndices{dim, NTuple{dim, UnitRange{Int}}}}}
    values::V
    indices::VI
end

# constructors
function MPValues{dim, T}(values::NamedTuple, indices::AbstractVector) where {dim, T}
    MPValuesBaseType = get_mpvalues_basetype(values.N, values.∇N)
    @assert MPValuesBaseType == MPValues{dim, T}
    MPValuesBaseType{typeof(values), typeof(indices)}(values, indices)
end
@generated function MPValues(info::MPValuesInfo{dim, T, <: NamedTuple{names}}, len::Int) where {dim, T, names}
    arrays = map(1:length(names)) do i
        name = names[i]
        dims = :((info.sizes[$i]..., len))
        :(fill(info.values.$name, $dims))
    end
    quote
        values = NamedTuple{names}(tuple($(arrays...)))
        indices = Vector{CartesianIndices{dim, NTuple{dim, UnitRange{Int}}}}(undef, len)
        MPValues{dim, T}(values, indices)
    end
end
# basically use these constructors
function MPValues{dim, T}(itp::Interpolation, len::Int) where {dim, T}
    info = MPValuesInfo{dim, T}(itp)
    MPValues(info, len)
end
MPValues{dim}(itp::Interpolation, len::Int) where {dim} = MPValues{dim, Float64}(itp, len)
get_mpvalues_basetype(N::AbstractArray{<: T}, ∇N::AbstractArray{<: Vec{dim, T}}) where {dim, T} = MPValues{dim, T}

# basic
Base.values(mps::MPValues) = getfield(mps, :values)
Base.propertynames(mps::MPValues) = propertynames(values(mps))
@inline Base.getproperty(mps::MPValues, name::Symbol) = getproperty(values(mps), name)

# getindex-like inferface
function Base.length(mps::MPValues)
    A = first(values(mps))
    size(A, ndims(A))
end
@inline function neighbornodes(mps::MPValues, i::Integer)
    @_propagate_inbounds_meta
    getfield(mps, :indices)[i]
end
@inline neighbornodes(mps::MPValues, ::Grid, i::Integer) = (@_propagate_inbounds_meta; neighbornodes(mps, i))
@inline neighbornodes(mps::MPValues, grid::SpGrid, i::Integer) = (@_propagate_inbounds_meta; nonzeroindices(get_sppat(grid), neighbornodes(mps, i)))
@inline function Base.values(mps::MPValues, i::Integer)
    @boundscheck @assert 1 ≤ i ≤ length(mps)
    SubMPValues(mps, i)
end
@inline function viewcol(A::AbstractArray, i::Integer)
    @boundscheck checkbounds(axes(A, ndims(A)), i)
    colons = nfill(:, Val(ndims(A)-1))
    @inbounds view(A, colons..., i)
end

struct SubMPValues{dim, T, V, VI, I}
    parent::MPValues{dim, T, V, VI}
    index::I
end

Base.parent(mp::SubMPValues) = getfield(mp, :parent)
Base.propertynames(mp::SubMPValues) = propertynames(parent(mp))

# `checkbounds` must already be done when constructing SubMPValues
# Then, there is no need to `checkbounds` because `SubMPValues` is
# immutable and its `parent` arrays can not be resized (because
# they are all multidimensional arrays)
@inline function Base.getproperty(mp::SubMPValues, name::Symbol)
    index = getfield(mp, :index)
    @inbounds viewcol(getproperty(parent(mp), name), index)
end
@inline function neighbornodes(mp::SubMPValues)
    index = getfield(mp, :index)
    @inbounds getfield(parent(mp), :indices)[index]
end
@inline function set_neighbornodes!(mp::SubMPValues, inds)
    index = getfield(mp, :index)
    @inbounds getfield(parent(mp), :indices)[index] = inds
end

###########
# update! #
###########

@inline getx(x::Vec) = x
@inline getx(pt) = @inbounds pt.x
@inline function alltrue(A::AbstractArray, indices::CartesianIndices)
    @boundscheck checkbounds(A, indices)
    @inbounds @simd for i in indices
        A[i] || return false
    end
    true
end
@inline function alltrue(A::Trues, indices::CartesianIndices)
    @boundscheck checkbounds(A, indices)
    true
end

# MPValues
function update!(mps::MPValues, itp::Interpolation, lattice::Lattice, sppat::AbstractArray{Bool}, particles::Particles)
    @assert length(mps) == length(particles)
    @assert size(lattice) == size(sppat)
    @threaded_inbounds for p in 1:length(mps)
        update!(values(mps, p), itp, lattice, sppat, LazyRow(particles, p))
    end
end
function update!(mps::MPValues, itp::Interpolation, lattice::Lattice, particles::Particles)
    update!(mps, itp, lattice, Trues(size(lattice)), particles)
end

# SubMPValues
@inline function update!(mp::SubMPValues, itp::Interpolation, lattice::Lattice, pt)
    update!(mp, itp, lattice, Trues(size(lattice)), pt)
end
@inline function update!(mp::SubMPValues, itp::Interpolation, lattice::Lattice, sppat::AbstractArray, pt)
    sppat isa Trues || @warn "Sparsity pattern on grid is not supported in `$itp`, just ignored" maxlog=1
    update!(mp, itp, lattice, pt)
end
