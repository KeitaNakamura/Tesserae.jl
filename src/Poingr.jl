module Poingr

using Base: @_inline_meta, @_propagate_inbounds_meta, @pure
using Base.Cartesian: @ntuple, @nall

using Base.Broadcast: Broadcasted, BroadcastStyle, AbstractArrayStyle, ArrayStyle, broadcasted, broadcastable, throwdm, preprocess

using Reexport
@reexport using Tensorial
@reexport using WriteVTK
using StaticArrays, StructArrays, FillArrays
using Coordinates

const BLOCK_UNIT = unsigned(3) # 2^3

export
# grid
    Grid,
    gridsteps,
    gridaxes,
    gridorigin,
    generate_pointstate,
    setbounds!,
    eachboundary,
# shape functions
    update!,
    BSpline,
    LinearBSpline,
    QuadraticBSpline,
    CubicBSpline,
    LinearWLS,
    BilinearWLS,
    polynomial,
    GIMP,
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
    SoilHypoelastic,
    SoilHyperelastic,
    VonMises,
    DruckerPrager,
    MonaghanWaterModel,
    NewtonianFluid,
    matcalc,
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


include("utils.jl")
include("dotmacros.jl")

include("sparray.jl")

abstract type ShapeFunction end
include("grid.jl")

include("ShapeFunctions/shapefunction.jl")
include("ShapeFunctions/bspline.jl")
include("ShapeFunctions/wls.jl")
include("ShapeFunctions/gimp.jl")

include("nodestate.jl")
include("pointstate.jl")

include("mpcache.jl")
include("contact_mechanics.jl")

abstract type MaterialModel end
Broadcast.broadcastable(x::MaterialModel) = (x,)
include("MaterialModels/utils.jl")
include("MaterialModels/LinearElastic.jl")
include("MaterialModels/SoilHypoelastic.jl")
include("MaterialModels/SoilHyperelastic.jl")
include("MaterialModels/VonMises.jl")
include("MaterialModels/DruckerPrager.jl")
include("MaterialModels/WaterModel.jl")
include("MaterialModels/NewtonianFluid.jl")

include("async.jl")

include("logger.jl")
include("vtk.jl")

end # module
