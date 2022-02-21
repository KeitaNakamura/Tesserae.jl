module Poingr

using Base: @_inline_meta, @_propagate_inbounds_meta, @pure
using Base.Cartesian: @ntuple, @nall

using Base.Broadcast: Broadcasted, BroadcastStyle, AbstractArrayStyle, ArrayStyle, broadcasted, broadcastable, throwdm, preprocess

using Reexport
@reexport using Tensorial
@reexport using WriteVTK
using StaticArrays, StructArrays

const BLOCK_UNIT = unsigned(3) # 2^3

export
# coordinate system
    CoordinateSystem,
    PlaneStrain,
    Axisymmetric,
# grid
    Grid,
    gridsteps,
    gridaxes,
    gridorigin,
    generate_pointstate,
    setbounds!,
    eachboundary,
# interpolations
    update!,
    BSpline,
    LinearBSpline,
    QuadraticBSpline,
    CubicBSpline,
    GIMP,
    LinearWLS,
    BilinearWLS,
    KernelCorrection,
    polynomial,
    MPValues,
# MPCache
    MPCache,
    point_to_grid!,
    grid_to_point!,
    grid_to_point,
# Transfer
    Transfer,
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

include("coordinate_system.jl")
include("sparray.jl")

abstract type Interpolation end
include("grid.jl")

include("Interpolations/mpvalues.jl")
include("Interpolations/bspline.jl")
include("Interpolations/gimp.jl")
include("Interpolations/basis.jl")
include("Interpolations/wls.jl")
include("Interpolations/correction.jl")

include("nodestate.jl")
include("pointstate.jl")

include("mpcache.jl")
include("transfer.jl")
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
