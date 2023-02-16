module Marble

using Base: RefValue, @_inline_meta, @_propagate_inbounds_meta, @pure
using Base.Broadcast: Broadcasted, ArrayStyle, DefaultArrayStyle
using Base.Cartesian: @ntuple, @nall, @nexprs

using Reexport
@reexport using Tensorial
@reexport using WriteVTK
using StructArrays

import SIMD
const SVec = SIMD.Vec
const SIMDTypes = Union{Float16, Float32, Float64}

# reexport from StructArrays
export LazyRow, LazyRows

const BLOCK_UNIT = unsigned(3) # 2^3

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
    spacing,
    neighbornodes,
# Grid
    Grid,
    SpGrid,
    generate_grid,
# Particles
    Particles,
    generate_particles,
# interpolations
    update!,
    num_nodes,
    shape_value,
    shape_gradient,
    Interpolation,
    BSpline,
    LinearBSpline,
    QuadraticBSpline,
    CubicBSpline,
    uGIMP,
    Kernel,
    LinearWLS,
    KernelCorrection,
    MPValue,
# MPSpace
    MPSpace,
    mpvalue,
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
include("Interpolations/mpvalue.jl")
include("Interpolations/bspline.jl")
include("Interpolations/gimp.jl")
include("Interpolations/polybasis.jl")
include("Interpolations/wls.jl")
include("Interpolations/kernelcorrection.jl")
include("mpspace.jl")
include("transfer.jl")

include("vtk.jl")

end # module
