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
MPValues{dim, T}(values::V, indices::CI) where {dim, T, V, CI} = MPValues{dim, T, V, CI}(values, indices)
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

# basic
Base.values(mps::MPValues) = getfield(mps, :values)
Base.propertynames(mps::MPValues) = propertynames(values(mps))
@inline Base.getproperty(mps::MPValues, name::Symbol) = getproperty(values(mps), name)

# getindex-like inferface
function Base.length(mps::MPValues{<: Any, <: Any})
    A = first(values(mps))
    size(A, ndims(A))
end
@inline function neighbornodes(mps::MPValues{<: Any, <: Any}, i::Integer)
    @_propagate_inbounds_meta
    getfield(mps, :indices)[i]
end
@inline function set_neighbornodes!(mps::MPValues{<: Any, <: Any}, i::Integer, inds)
    @_propagate_inbounds_meta
    getfield(mps, :indices)[i] = inds
end
@generated function Base.values(mps::MPValues{dim, T, <: NamedTuple{names}}, i::Integer) where {dim, T, names}
    exps = [:(viewcol(mps.$name, i)) for name in names]
    quote
        @_propagate_inbounds_meta
        values = NamedTuple{names}(tuple($(exps...)))
        indices = neighbornodes(mps, i)
        SubMPValues{dim, T}(values, indices)
    end
end
@inline function viewcol(A::AbstractArray, i::Integer)
    @boundscheck checkbounds(axes(A, ndims(A)), i)
    colons = nfill(:, Val(ndims(A)-1))
    @inbounds view(A, colons..., i)
end

struct SubMPValues{dim, T, V <: NamedTuple}
    values::V
    indices::CartesianIndices{dim, NTuple{dim, UnitRange{Int}}}
end

SubMPValues{dim, T}(values::V, indices) where {dim, T, V} = SubMPValues{dim, T, V}(values, indices)

Base.values(mps::SubMPValues) = getfield(mps, :values)
Base.propertynames(mps::SubMPValues) = propertynames(values(mps))
@inline Base.getproperty(mps::SubMPValues, name::Symbol) = getproperty(values(mps), name)
@inline neighbornodes(mps::SubMPValues) = getfield(mps, :indices)

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

# isparent=true
function update!(mps::MPValues, itp::Interpolation, lattice::Lattice, sppat::AbstractArray{Bool}, particles::Particles)
    @assert length(mps) == length(particles)
    @assert size(lattice) == size(sppat)
    @threaded for p in 1:length(mps)
        indices = update!(values(mps, p), itp, lattice, sppat, LazyRow(particles, p))
        set_neighbornodes!(mps, p, indices)
    end
end
function update!(mps::MPValues, itp::Interpolation, lattice::Lattice, particles::Particles)
    update!(mps, itp, lattice, Trues(size(lattice)), particles)
end

# isparent=false
@inline function update!(mp::SubMPValues, itp::Interpolation, lattice::Lattice, pt)
    indices = update_mpvalues!(mp, itp, lattice, pt)
    indices isa CartesianIndices || error("`update_mpvalues` must return `CartesianIndices`")
    indices
end
@inline function update!(mp::SubMPValues, itp::Interpolation, lattice::Lattice, sppat::AbstractArray{Bool}, pt)
    @assert size(lattice) == size(sppat)
    indices = update_mpvalues!(mp, itp, lattice, sppat, pt)
    indices isa CartesianIndices || error("`update_mpvalues` must return `CartesianIndices`")
    indices
end

@inline function update_mpvalues!(mp::SubMPValues, itp::Interpolation, lattice::Lattice, pt)
    update_mpvalues!(mp, itp, lattice, Trues(size(lattice)), pt)
end
@inline function update_mpvalues!(mp::SubMPValues, itp::Interpolation, lattice::Lattice, sppat::AbstractArray, pt)
    sppat isa Trues || @warn "Sparsity pattern on grid is not supported in `$(typeof(mp))`, just ignored" maxlog=1
    update_mpvalues!(mp, itp, lattice, pt)
end
