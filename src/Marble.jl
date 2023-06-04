module Marble

using Base: @_inline_meta, @_propagate_inbounds_meta
using Base.Broadcast: Broadcasted, ArrayStyle, DefaultArrayStyle
using Base.Cartesian: @ntuple, @nall, @nexprs

using Reexport
using LinearAlgebra
@reexport using Tensorial
using StructArrays

# SIMD
import SIMD
const SVec = SIMD.Vec
const SIMDTypes = Union{SIMD.ScalarTypes, Bool}

# stream
using WriteVTK
import ProgressMeter

# reexport from StructArrays
export LazyRow, LazyRows

export
# utils
    fillzero!,
    @rename,
    flatarray,
# coordinate system
    CoordinateSystem,
    DefaultSystem,
    PlaneStrain,
    Axisymmetric,
# SpArray
    SpArray,
    blocksize,
    blocksparsity,
    update_sparsity!,
# lattice
    Lattice,
# Grid
    Grid,
    SpGrid,
    eachnode,
    generate_grid,
    spacing,
    isnonzero,
# Particles
    Particles,
    eachparticle,
    generate_particles,
    GridSampling,
    PoissonDiskSampling,
    BoxDomain,
    SphericalDomain,
    FunctionDomain,
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
    FLIP,
    PIC,
    AffineTransfer,
    AFLIP,
    APIC,
    TaylorTransfer,
    TFLIP,
    TPIC,
    WLSTransfer,
# implicit
    NewtonMethod,
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
include("implicit.jl")

include("vtk.jl")

end # module
