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
    isnearbounds::Vector{Bool}
end

# constructors
function MPValues(values::NamedTuple, indices::AbstractVector, isnearbounds::Vector{Bool})
    MPValuesBaseType = get_mpvalues_basetype(values.N, values.∇N)
    MPValuesBaseType{typeof(values), typeof(indices)}(values, indices, isnearbounds)
end
@generated function MPValues(info::MPValuesInfo{dim, T, <: NamedTuple{names}}, len::Int) where {dim, T, names}
    arrays = map(1:length(names)) do i
        name = names[i]
        dims = :((info.sizes[$i]..., len))
        :(fill(info.values.$name, $dims))
    end
    quote
        values = NamedTuple{names}(tuple($(arrays...)))
        indices = fill(CartesianIndices(nfill(0:0, Val(dim))), len)
        MPValues(values, indices, fill(false, len))
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
Base.values(mpvalues::MPValues) = getfield(mpvalues, :values)
Base.propertynames(mpvalues::MPValues) = propertynames(values(mpvalues))
@inline Base.getproperty(mpvalues::MPValues, name::Symbol) = getproperty(values(mpvalues), name)

# values
function num_particles(mpvalues::MPValues)
    A = first(values(mpvalues))
    size(A, ndims(A))
end
@inline function Base.values(mpvalues::MPValues, p::Integer)
    @boundscheck @assert 1 ≤ p ≤ num_particles(mpvalues)
    SubMPValues(mpvalues, p)
end
@inline function viewcol(A::AbstractArray, i::Integer)
    @boundscheck checkbounds(axes(A, ndims(A)), i)
    colons = nfill(:, Val(ndims(A)-1))
    @inbounds view(A, colons..., i)
end

function Base.show(io::IO, mpvalues::MPValues{dim, T}) where {dim, T}
    print(io, "MPValues{$dim, $T}: \n")
    print(io, "  Particles: ", commas(num_particles(mpvalues)), "\n")
    print(io, "  Storage size: ", size(mpvalues.N))
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
@inline function neighbornodes(mp::SubMPValues, grid::Grid)
    inds = neighbornodes(mp)
    @boundscheck checkbounds(grid, inds)
    inds
end
@inline function neighbornodes(mp::SubMPValues, grid::SpGrid)
    inds = neighbornodes(mp)
    @boundscheck checkbounds(grid, inds)
    @inbounds nonzeroindices(get_spinds(grid), inds)
end
@inline function isnearbounds(mp::SubMPValues)
    index = getfield(mp, :index)
    @inbounds getfield(parent(mp), :isnearbounds)[index]
end
@inline function set_neighbornodes!(mp::SubMPValues, inds)
    index = getfield(mp, :index)
    @inbounds getfield(parent(mp), :indices)[index] = inds
end
@inline function set_isnearbounds!(mp::SubMPValues, isnearbounds)
    index = getfield(mp, :index)
    @inbounds getfield(parent(mp), :isnearbounds)[index] = isnearbounds
end

function Base.show(io::IO, mp::SubMPValues)
    print(io, "SubMPValues: \n")
    print(io, "  Array size: ", size(mp.N), "\n")
    print(io, "  Property names: ")
    print(io, join(map(propertynames(mp)) do name
        string(name, "::", eltype(typeof(getproperty(mp, name))))
    end, ", "), "\n")
    print(io, "  Neighbor nodes: ", neighbornodes(mp), "\n")
    print(io, "  Is near bounds: ", isnearbounds(mp))
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
function update!(mpvalues::MPValues, itp::Interpolation, lattice::Lattice, spy::AbstractArray{Bool}, particles::Particles; parallel::Bool)
    @assert num_particles(mpvalues) == length(particles)
    @assert size(lattice) == size(spy)
    @threads_inbounds parallel for p in 1:num_particles(mpvalues)
        update!(values(mpvalues, p), itp, lattice, spy, LazyRow(particles, p))
    end
end
function update!(mpvalues::MPValues, itp::Interpolation, lattice::Lattice, particles::Particles; parallel::Bool)
    update!(mpvalues, itp, lattice, Trues(size(lattice)), particles; parallel)
end

# SubMPValues
@inline function update!(mp::SubMPValues, itp::Interpolation, lattice::Lattice, spy::AbstractArray{Bool}, pt)
    indices = neighbornodes(itp, lattice, pt)
    isfullyinside = size(getproperty(mp, first(propertynames(mp)))) == size(indices)
    isnearbounds = !isfullyinside || !(@inbounds alltrue(spy, indices))
    set_neighbornodes!(mp, indices)
    set_isnearbounds!(mp, isnearbounds)
    update_mpvalues!(mp, itp, lattice, spy, pt)
end
@inline update!(mp::SubMPValues, itp::Interpolation, lattice::Lattice, pt) = update!(mp, itp, lattice, Trues(size(lattice)), pt)

@inline function update_mpvalues!(mp::SubMPValues, itp::Interpolation, lattice::Lattice, pt)
    update!(mp, itp, lattice, Trues(size(lattice)), pt)
end
@inline function update_mpvalues!(mp::SubMPValues, itp::Interpolation, lattice::Lattice, spy::AbstractArray, pt)
    spy isa Trues || @warn "Sparsity pattern on grid is not supported in `$itp`, just ignored" maxlog=1
    update_mpvalues!(mp, itp, lattice, pt)
end
