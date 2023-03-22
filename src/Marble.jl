module Marble

using Base: @_inline_meta, @_propagate_inbounds_meta
using Base.Broadcast: Broadcasted, ArrayStyle, DefaultArrayStyle
using Base.Cartesian: @ntuple, @nall, @nexprs

using Reexport
@reexport using Tensorial
@reexport using WriteVTK
using StructArrays

# SIMD
import SIMD
const SVec = SIMD.Vec
const SIMDTypes = Union{Float16, Float32, Float64}

# reexport from StructArrays
export LazyRow, LazyRows

export
# utils
    fillzero!,
    @rename,
# coordinate system
    CoordinateSystem,
    PlaneStrain,
    Axisymmetric,
# lattice
    Lattice,
# Grid
    Grid,
    SpGrid,
    generate_grid,
    spacing,
    isnonzero,
# Particles
    Particles,
    generate_particles,
# interpolations
    update!,
    Kernel,
    BSpline,
    LinearBSpline,
    QuadraticBSpline,
    CubicBSpline,
    uGIMP,
    Interpolation,
    LinearWLS,
    KernelCorrection,
    MPValues,
# MPSpace
    MPSpace,
    num_particles,
    neighbornodes,
# Transfer
    particle_to_grid!,
    grid_to_particle!,
    TransferAlgorithm,
    DefaultTransfer,
    FLIP,
    PIC,
    AffineTransfer,
    AFLIP,
    APIC,
    TaylorTransfer,
    TFLIP,
    TPIC,
# VTK
    openvtk,
    openvtm,
    openpvd,
    closevtk,
    closevtm,
    closepvd


include("utils.jl")
include("sparray.jl")
include("lattice.jl")

include("grid.jl")
include("particles.jl")
include("Interpolations/mpvalues.jl")
include("Interpolations/bspline.jl")
include("Interpolations/gimp.jl")
include("Interpolations/polybasis.jl")
include("Interpolations/wls.jl")
include("Interpolations/kernelcorrection.jl")
include("blockspace.jl")
include("mpspace.jl")
include("transfer.jl")

include("vtk.jl")

end # module
