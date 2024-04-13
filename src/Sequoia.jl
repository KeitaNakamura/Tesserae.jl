module Sequoia

using Base: @_inline_meta, @_propagate_inbounds_meta
using Base.Broadcast: Broadcasted, ArrayStyle, DefaultArrayStyle
using Base.Cartesian: @ntuple, @nall, @nexprs

using SparseArrays

using Reexport
@reexport using Tensorial
using Tensorial: resizedim
export resizedim

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
    @threaded,
# BlockSpace
    BlockSpace,
    blocksize,
# SpArray
    SpArray,
    update_block_sparsity!,
# Mesh
    CartesianMesh,
    volume,
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
    GIMP,
    Interpolation,
    WLS,
    KernelCorrection,
# MPValue
    neighboringnodes,
    MPValue,
    MPValueVector,
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
include("mesh.jl")
include("blockspace.jl")
include("sparray.jl")

include("grid.jl")
include("particles.jl")

include("Interpolations/mpvalue.jl")
include("Interpolations/bspline.jl")
include("Interpolations/gimp.jl")
include("Interpolations/polynomials.jl")
include("Interpolations/wls.jl")
include("Interpolations/kernelcorrection.jl")

include("multigridspace.jl")

include("transfer.jl")
include("implicit.jl")

include("vtk.jl")

end # module Sequoia
