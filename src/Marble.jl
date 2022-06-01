module Marble

using Base: @_inline_meta, @_propagate_inbounds_meta, @pure
using Base.Broadcast: Broadcasted, ArrayStyle, broadcasted

using Reexport
@reexport using Tensorial
@reexport using WriteVTK
using StaticArrays, StructArrays

const BLOCK_UNIT = unsigned(3) # 2^3

export
# dot macros
    @dot_threads,
    @dot_lazy,
# coordinate system
    CoordinateSystem,
    PlaneStrain,
    Axisymmetric,
# grid
    Grid,
    gridsteps,
    gridaxes,
    gridorigin,
    boundaries,
    generate_pointstate,
# interpolations
    update!,
    Interpolation,
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
    TransferNormalFLIP,
    TransferNormalPIC,
    TransferTaylorFLIP,
    TransferTaylorPIC,
    TransferAffinePIC,
# Frictional contact
    CoulombFriction,
    contacted,
# Logger
    Logger,
    isfinised,
    islogpoint,
    logindex,
# VTK
    vtk_points


include("utils.jl")
include("dotmacros.jl")
include("sparray.jl")
include("misc.jl")

abstract type Interpolation end
include("grid.jl")

include("Interpolations/mpvalues.jl")
include("Interpolations/bspline.jl")
include("Interpolations/gimp.jl")
include("Interpolations/basis.jl")
include("Interpolations/wls.jl")
include("Interpolations/correction.jl")

include("states.jl")
include("mpcache.jl")
include("transfer.jl")
include("contact.jl")

include("logger.jl")
include("vtk.jl")

end # module
