module Tesserae

using Base: @propagate_inbounds, @_inline_meta, @_propagate_inbounds_meta
using Base.Broadcast: Broadcasted, ArrayStyle, DefaultArrayStyle
using Base.Cartesian: @ntuple, @nall, @nexprs

using SparseArrays

using Reexport
@reexport using Tensorial
using Tensorial: resizedim
export resizedim

using StructArrays

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
# BlockSpace
    BlockSpace,
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
# Particles
    generate_particles,
    GridSampling,
    PoissonDiskSampling,
# interpolations
    update!,
    BSpline,
    LinearBSpline,
    QuadraticBSpline,
    CubicBSpline,
    SteffenBSpline,
    SteffenLinearBSpline,
    SteffenQuadraticBSpline,
    SteffenCubicBSpline,
    uGIMP,
    Interpolation,
    WLS,
    KernelCorrection,
# MPValue
    generate_mpvalues,
    neighboringnodes,
    MPValue,
# transfer
    @P2G,
    @G2P,
# implicit
    DofMap,
    ndofs,
    create_sparse_matrix,
    submatrix,
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

include("transfer.jl")
include("implicit.jl")

include("vtk.jl")

end # module Tesserae
