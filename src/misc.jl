#########
# Index #
#########

struct Index{dim}
    i::Int
    I::CartesianIndex{dim}
end
@inline Base.checkindex(::Type{Bool}, inds::AbstractUnitRange, i::Index) = checkindex(Bool, inds, i.i)
@inline _to_indices(::IndexLinear, A, inds, I::Tuple{Index, Vararg{Any}}) = to_indices(A, inds, (I[1].i, Base.tail(I)...))
@inline _to_indices(::IndexCartesian, A, inds, I::Tuple{Index, Vararg{Any}}) = to_indices(A, inds, (Tuple(I[1].I)..., Base.tail(I)...))
@inline Base.to_indices(A, inds, I::Tuple{Index, Vararg{Any}}) = _to_indices(IndexStyle(A), A, inds, I)

####################
# CoordinateSystem #
####################

abstract type CoordinateSystem end

struct OneDimensional   <: CoordinateSystem end
struct PlaneStrain      <: CoordinateSystem end
struct Axisymmetric     <: CoordinateSystem end
struct ThreeDimensional <: CoordinateSystem end

# default coordinate system
get_coordinate_system(::Nothing, ::Val{1}) = OneDimensional()
get_coordinate_system(::Nothing, ::Val{2}) = PlaneStrain()
get_coordinate_system(::Nothing, ::Val{3}) = ThreeDimensional()

# check coordinate system
get_coordinate_system(::PlaneStrain, ::Val{2}) = PlaneStrain()
get_coordinate_system(::Axisymmetric, ::Val{2}) = Axisymmetric()
get_coordinate_system(c::CoordinateSystem, ::Val{dim}) where {dim} = 
    throw(ArgumentError("wrong coordinate system $c for dimension $dim"))
