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
    @foreach,
    @showprogress,
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
    cells,
    boundaries,
    IGAPatch,
    IGACell,
    IGAMesh,
    IGABasis,
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
    @explain,
    ExplainedCode,
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
    readmsh,
# GPU
    cpu,
    gpu,
    gpu_preserve

include("devices.jl")
include("utils.jl")
include("progress.jl")
import .Progress: @showprogress

include("shapes.jl")
include("NURBS/NURBS.jl")

include("mesh.jl")
include("igamesh.jl")
include("thread_partition.jl")
include("sparray.jl")
include("hybridarray.jl")

include("grid.jl")
include("particles.jl")

include("Basis/basis_weight.jl")
include("Basis/iga.jl")
include("Basis/bspline.jl")
include("Basis/gimp.jl")
include("Basis/cpdi.jl")
include("Basis/polynomials.jl")
include("Basis/wls.jl")
include("Basis/kernelcorrection.jl")

include("fem.jl")
include("gmsh.jl")

include("transfer.jl")
include("foreach.jl")
include("implicit.jl")
include("explain.jl")

include("export.jl")
include("gpu.jl")
include("deprecated.jl")

include("Stencil/Stencil.jl")

end # module Tesserae
