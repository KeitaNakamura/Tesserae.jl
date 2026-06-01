module Tesserae

using Base: @propagate_inbounds, @_inline_meta, @_propagate_inbounds_meta
using Base.Broadcast: Broadcasted, ArrayStyle
using Base.Cartesian: @ntuple, @nall, @nexprs

using SparseArrays
using Printf

using Reexport
@reexport using Tensorial

using StaticArrays
using StructArrays

# sampling
import PoissonDiskSampling: generate as poisson_disk_sampling
import Random

# others
import Preferences
import ProgressMeter

# multithreading
using Graphs
using TaskLocalValues

# GPU
using GPUArraysCore
using KernelAbstractions
using Adapt
using Atomix

export
# utils
    fillzero!,
    @threaded,
# ThreadPartition
    ThreadPartition,
    ColorPartition,
    threadsafe_groups,
    particle_indices,
# SpArray
    SpArray,
    update_sparsity!,
# Mesh
    CartesianMesh,
    volume,
    isinside,
    findcell,
    extract,
    UnstructuredMesh,
# Grid
    Grid,
    SpGrid,
    generate_grid,
    spacing,
# Particles
    generate_particles,
    reorder_particles!,
    GridSampling,
    PoissonDiskSampling,
# basis functions
    update!,
    Order,
    Constant,
    Linear,
    Quadratic,
    Cubic,
    MultiLinear,
    Polynomial,
    BSpline,
    SteffenBSpline,
    uGIMP,
    CPDI,
    Basis,
    Interpolation,
    WLS,
    KernelCorrection,
# BasisWeight
    generate_basis_weights,
    generate_interpolation_weights,
    neighboringnodes,
    supportnodes,
    basis,
    BasisWeight,
    BasisWeightArray,
    InterpolationWeight,
    feupdate!,
# transfer
    @P2G,
    @G2P,
    @G2P2G,
# implicit
    DofMap,
    ndofs,
    dofs,
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
    gpu,
    gpu_preserve

include("devices.jl")
include("utils.jl")

include("shapes.jl")

include("mesh.jl")
include("partitioning.jl")
include("sparray.jl")
include("hybridarray.jl")

include("grid.jl")
include("particles.jl")

include("Basis/basis_weight.jl")
include("Basis/bspline.jl")
include("Basis/gimp.jl")
include("Basis/cpdi.jl")
include("Basis/polynomials.jl")
include("Basis/wls.jl")
include("Basis/kernelcorrection.jl")

include("fem.jl")

include("transfer.jl")
include("implicit.jl")

include("export.jl")
include("gpu.jl")
include("deprecated.jl")

include("Stencil/Stencil.jl")

end # module Tesserae
