module Sequoia

using Base: @_inline_meta, @_propagate_inbounds_meta
using Base.Broadcast: Broadcasted, ArrayStyle, DefaultArrayStyle
using Base.Cartesian: @ntuple, @nall, @nexprs

using SparseArrays

using Reexport
@reexport using Tensorial

using StructArrays
export LazyRow, LazyRows

# sampling
import PoissonDiskSampling: generate as poisson_disk_sampling
import Random

# stream
using WriteVTK

# others
import Preferences
import ProgressMeter

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
    isinside,
    whichcell,
# Grid
    Grid,
    SpGrid,
    generate_grid,
    spacing,
    spacing_inv,
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
    surroundingnodes,
    MPValues,
    MPValuesVector,
# transfer
    @P2G,
    @G2P,
# implicit
    DofMap,
    ndofs,
    create_sparse_matrix,
    getsubset,
    @P2G_Matrix,
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
include("Interpolations/polynomials.jl")
include("Interpolations/wls.jl")
include("Interpolations/kernelcorrection.jl")

include("transfer.jl")
include("implicit.jl")

include("vtk.jl")

end # module Sequoia
