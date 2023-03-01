abstract type Interpolation end
abstract type Kernel <: Interpolation end

Broadcast.broadcastable(interp::Interpolation) = (interp,)

# used for constructing `MPValues`
struct InterpolationInfo{dim, T, V <: NamedTuple, S <: Tuple{Vararg{Dims}}}
    values::V
    sizes::S
end
function InterpolationInfo{dim, T}(values::NamedTuple, sizes::Tuple{Vararg{Dims}}) where {dim, T}
    @assert length(values) == length(sizes)
    InterpolationInfo{dim, T, typeof(values), typeof(sizes)}(values, sizes)
end

"""
    MPValues{dim}(interpolation, length)
    MPValues{dim, T}(interpolation, length)
"""
struct MPValues{dim, T, isparent, V <: NamedTuple}
    values::V
end

################
# constructors #
################

MPValues{dim, T, isparent}(values::V) where {dim, T, isparent, V} = MPValues{dim, T, isparent::Bool, V}(values)

@generated function MPValues(info::InterpolationInfo{dim, T, <: NamedTuple{names}}, len::Int) where {dim, T, names}
    arrays = map(1:length(names)) do i
        name = names[i]
        dims = :((info.sizes[$i]..., len))
        :(fill(info.values.$name, $dims))
    end
    quote
        values = NamedTuple{names}(tuple($(arrays...)))
        MPValues{dim, T, true}(values)
    end
end

# use these constructors
function MPValues{dim, T}(itp::Interpolation, len::Int) where {dim, T}
    info = InterpolationInfo{dim, T}(itp)
    MPValues(info, len)
end
MPValues{dim}(itp::Interpolation, len::Int) where {dim} = MPValues{dim, Float64}(itp, len)

###########
# methods #
###########

Base.values(mps::MPValues) = getfield(mps, :values)
Base.propertynames(mps::MPValues) = propertynames(values(mps))
@inline Base.getproperty(mps::MPValues, name::Symbol) = getproperty(values(mps), name)

# getindex inferface for `isparent=true`
function Base.length(mps::MPValues{<: Any, <: Any, true})
    A = first(values(mps))
    size(A, ndims(A))
end
@generated function Base.getindex(mps::MPValues{dim, T, true, <: NamedTuple{names}}, i::Integer) where {dim, T, names}
    exps = [:(_getarray(mps.$name, i)) for name in names]
    quote
        @_propagate_inbounds_meta
        values = NamedTuple{names}(tuple($(exps...)))
        MPValues{dim, T, false}(values)
    end
end
@generated function _getarray(arr::AbstractArray{<: Any, N}, i::Integer) where {N}
    colons = fill(:, N-1)
    quote
        @_inline_meta
        @boundscheck checkbounds(axes(arr, N), i)
        @inbounds view(arr, $(colons...), i)
    end
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

@inline function update!(mp::MPValues{<: Any, <: Any, false}, itp::Interpolation, lattice::Lattice, pt)
    indices = update_mpvalues!(mp, itp, lattice, pt)
    indices isa CartesianIndices || error("`update_mpvalues` must return `CartesianIndices`")
    indices
end
@inline function update!(mp::MPValues{<: Any, <: Any, false}, itp::Interpolation, lattice::Lattice, sppat::AbstractArray{Bool}, pt)
    @assert size(lattice) == size(sppat)
    indices = update_mpvalues!(mp, itp, lattice, sppat, pt)
    indices isa CartesianIndices || error("`update_mpvalues` must return `CartesianIndices`")
    indices
end

@inline function update_mpvalues!(mp::MPValues, itp::Interpolation, lattice::Lattice, pt)
    update_mpvalues!(mp, itp, lattice, Trues(size(lattice)), pt)
end
@inline function update_mpvalues!(mp::MPValues, itp::Interpolation, lattice::Lattice, sppat::AbstractArray, pt)
    sppat isa Trues || @warn "Sparsity pattern on grid is not supported in `$(typeof(mp))`, just ignored" maxlog=1
    update_mpvalues!(mp, itp, lattice, pt)
end
