module Sequoia

using Base: @_inline_meta, @_propagate_inbounds_meta
using Base.Broadcast: Broadcasted, ArrayStyle, DefaultArrayStyle
using Base.Cartesian: @ntuple, @nall, @nexprs

using Reexport
@reexport using Tensorial
using Tensorial: resizedim
export resizedim

using StructArrays
export LazyRow, LazyRows

# SIMD
import SIMD
const SVec = SIMD.Vec
const SIMDTypes = Union{SIMD.ScalarTypes, Bool}

# sampling
import PoissonDiskSampling: generate as poisson_disk_sampling
import Random

# stream
using WriteVTK

# others
import Preferences

export
# utils
    fillzero!,
    @threaded,
# SpArray/SpSpace
    SpArray,
    SpSpace,
    blocksize,
    update_block_sparsity!,
# lattice
    Lattice,
# Grid
    Grid,
    SpGrid,
    generate_grid,
    spacing,
    isactive,
# Particles
    generate_particles,
    GridSampling,
    PoissonDiskSampling,
# interpolations
    interpolation,
    update!,
    BSpline,
    LinearBSpline,
    QuadraticBSpline,
    CubicBSpline,
    uGIMP,
    Interpolation,
    WLS,
    KernelCorrection,
# MPValues
    neighbornodes,
    MPValues,
    MPValuesVector,
# transfer
    @P2G,
    @G2P,
# VTK
    openvtk,
    openvtm,
    openpvd,
    closevtk,
    closevtm,
    closepvd

include("utils.jl")
include("lattice.jl")
include("sparray.jl")
include("spspace.jl")

include("grid.jl")
include("particles.jl")

include("Interpolations/mpvalues.jl")
include("Interpolations/bspline.jl")
include("Interpolations/gimp.jl")
include("Interpolations/wls.jl")
include("Interpolations/kernelcorrection.jl")

include("transfer.jl")

include("vtk.jl")

end # module Sequoia
