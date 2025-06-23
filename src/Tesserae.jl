module Tesserae

using Base: @propagate_inbounds, @_inline_meta, @_propagate_inbounds_meta
using Base.Broadcast: Broadcasted, ArrayStyle, DefaultArrayStyle
using Base.Cartesian: @ntuple, @nall, @nexprs

using SparseArrays

using Reexport
@reexport using Tensorial
using Tensorial: ∂ⁿ

using StaticArrays
using StructArrays

# sampling
import PoissonDiskSampling: generate as poisson_disk_sampling
import Random

# others
import Preferences
import ProgressMeter

# GPU
using GPUArraysCore
using KernelAbstractions
using Adapt
using Atomix

export
# utils
    fillzero!,
    @threaded,
# ColorPartition
    ColorPartition,
# SpArray
    SpArray,
    update_block_sparsity!,
# Mesh
    CartesianMesh,
    volume,
    isinside,
    whichcell,
    extract,
    UnstructuredMesh,
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
    Order,
    Linear,
    Quadratic,
    Cubic,
    Quartic,
    Quintic,
    MultiLinear,
    MultiQuadratic,
    Polynomial,
    BSpline,
    SteffenBSpline,
    uGIMP,
    CPDI,
    Interpolation,
    WLS,
    KernelCorrection,
# MPValue
    generate_mpvalues,
    neighboringnodes,
    MPValue,
    feupdate!,
# transfer
    @P2G,
    @G2P,
    @G2P2G,
# implicit
    DofMap,
    ndofs,
    create_sparse_matrix,
    @P2G_Matrix,
# VTK
    openvtk,
    openvtm,
    openpvd,
    closevtk,
    closevtm,
    closepvd,
# GPU
    cpu,
    gpu

include("devices.jl")
include("utils.jl")

include("shapes.jl")

include("mesh.jl")
include("partitioning.jl")
include("sparray.jl")

include("grid.jl")
include("particles.jl")

include("Interpolations/mpvalue.jl")
include("Interpolations/bspline.jl")
include("Interpolations/gimp.jl")
include("Interpolations/cpdi.jl")
include("Interpolations/polynomials.jl")
include("Interpolations/wls.jl")
include("Interpolations/kernelcorrection.jl")

include("fem.jl")

include("transfer.jl")
include("implicit.jl")

include("export.jl")

include("gpu.jl")

end # module Tesserae
