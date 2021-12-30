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
