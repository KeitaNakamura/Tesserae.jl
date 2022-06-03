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
    gridbounds,
    generate_gridstate,
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
    MPValues,
# MPSpace
    MPSpace,
    point_to_grid!,
    grid_to_point!,
    grid_to_point,
    update_sparsitypattern!,
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


include("misc.jl")
include("utils.jl")
include("dotmacros.jl")
include("sparray.jl")

# core
include("grid.jl")
include("Interpolations/mpvalues.jl")
include("Interpolations/bspline.jl")
include("Interpolations/gimp.jl")
include("Interpolations/basis.jl")
include("Interpolations/wls.jl")
include("Interpolations/correction.jl")
include("mpspace.jl")

include("states.jl")
include("transfer.jl")
include("contact.jl")

# io
include("logger.jl")
include("vtk.jl")

end # module
