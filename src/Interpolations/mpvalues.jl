abstract type Interpolation end
abstract type Kernel <: Interpolation end

"""
    MPValues(Vec{dim}, interpolation)
    MPValues(Vec{dim, T}, interpolation)
"""
struct MPValues{It, Prop, Indices}
    it::It
    prop::Prop
    indices::Base.RefValue{Indices}
end

function MPValues(::Type{Vec{dim, T}}, it::Interpolation) where {dim, T}
    prop = create_property(Vec{dim, T}, it)
    indices = CartesianIndices(nfill(0:0, Val(dim)))
    MPValues(it, prop, Ref(indices))
end
MPValues(::Type{Vec{dim}}, it::Interpolation) where {dim} = MPValues(Vec{dim, Float64}, it)

Base.propertynames(mp::MPValues) = propertynames(getfield(mp, :prop))
@inline function Base.getproperty(mp::MPValues, name::Symbol)
    getproperty(getfield(mp, :prop), name)
end
@inline function Base.setproperty!(mp::MPValues, name::Symbol, v)
    setproperty!(getfield(mp, :prop), name, v)
end

@inline interpolation(mp::MPValues) = getfield(mp, :it)

@inline neighbornodes(mp::MPValues) = getfield(mp, :indices)[]
@inline function neighbornodes(mp::MPValues, grid::Grid)
    inds = neighbornodes(mp)
    @boundscheck checkbounds(grid, inds)
    inds
end
@inline function neighbornodes(mp::MPValues, grid::SpGrid)
    inds = neighbornodes(mp)
    spinds = get_spinds(grid)
    @boundscheck checkbounds(spinds, inds)
    @inbounds neighbors = view(spinds, inds)
    @boundscheck all(isactive, neighbors)
    neighbors
end

@inline function set_neighbornodes!(mp::MPValues, indices)
    getfield(mp, :indices)[] = indices
    mp
end

function Base.show(io::IO, mp::MPValues)
    print(io, "MPValues: \n")
    print(io, "  Interpolation: ", interpolation(mp), "\n")
    print(io, "  Property names: ")
    print(io, join(map(propertynames(mp)) do name
        string(name, "::", typeof(getproperty(mp, name)))
    end, ", "), "\n")
    print(io, "  Neighbor nodes: ", neighbornodes(mp))
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

function update!(mp::MPValues, lattice, pt)
    set_neighbornodes!(mp, neighbornodes(interpolation(mp), lattice, pt))
    update_property!(mp, lattice, pt)
end

function update!(mp::MPValues, lattice, pt, filter)
    @debug @assert size(lattice) == size(filter)
    set_neighbornodes!(mp, neighbornodes(interpolation(mp), lattice, pt))
    update_property!(mp, lattice, pt, filter)
end
