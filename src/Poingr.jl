module Poingr

using Base: @_inline_meta, @_propagate_inbounds_meta
using Base.Cartesian: @ntuple, @nall

using Base.Broadcast: Broadcasted, BroadcastStyle, AbstractArrayStyle, ArrayStyle, broadcasted, broadcastable, throwdm, preprocess

using Reexport
@reexport using Tensorial
@reexport using WriteVTK
using StaticArrays, StructArrays
using Coordinates

const BLOCK_UNIT = unsigned(3) # 2^3

export
# CoordinateSystem
    PlaneStrain,
    Axisymmetric,
# grid
    Grid,
    gridsteps,
    gridaxes,
    gridorigin,
    generate_pointstate,
    eachboundary,
# shape functions
    update!,
    BSpline,
    LinearBSpline,
    QuadraticBSpline,
    CubicBSpline,
    WLS,
    polynomial,
# MPCache
    MPCache,
    point_to_grid!,
    grid_to_point!,
    default_point_to_grid!,
    default_grid_to_point!,
# Contact
    Contact,
# MaterialModel
    MaterialModel,
    LinearElastic,
    SoilElastic,
    VonMises,
    DruckerPrager,
    WaterModel,
    NewtonianFluid,
    update_stress,
    soundspeed,
    volumetric_stress,
    deviatoric_stress,
    volumetric_strain,
    deviatoric_strain,
# Logger
    Logger,
    isfinised,
    islogpoint,
    logindex,
# VTK
    vtk_points,
# async
    AsyncScheduler,
    currenttime,
    issynced,
    synced_pointstate,
    updatetimestep!,
    asyncstep!,
# dot macros
    @dot_threads,
    @dot_lazy


abstract type CoordinateSystem end
struct DefaultCoordinateSystem <: CoordinateSystem end
struct PlaneStrain <: CoordinateSystem end
struct Axisymmetric <: CoordinateSystem end

include("utils.jl")

include("ShapeFunctions/shapefunction.jl")
include("ShapeFunctions/bspline.jl")
include("ShapeFunctions/wls.jl")

include("sparray.jl")
include("grid.jl")
include("pointstate.jl")

include("mpcache.jl")
include("contact_mechanics.jl")

abstract type MaterialModel end
Broadcast.broadcastable(x::MaterialModel) = (x,)
include("MaterialModels/utils.jl")
include("MaterialModels/LinearElastic.jl")
include("MaterialModels/SoilElastic.jl")
include("MaterialModels/VonMises.jl")
include("MaterialModels/DruckerPrager.jl")
include("MaterialModels/WaterModel.jl")
include("MaterialModels/NewtonianFluid.jl")

include("async.jl")
include("dotmacros.jl")

include("logger.jl")
include("vtk.jl")

end # module
