module Tesserae

using Base: @propagate_inbounds, @_inline_meta, @_propagate_inbounds_meta
using Base.Broadcast: Broadcasted, ArrayStyle
using Base.Cartesian: @ntuple, @nall, @nexprs

using SparseArrays
using LinearAlgebra: SymTridiagonal, eigen
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
using MacroTools
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
    FEMesh,
    generate_field_meshes,
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
# Quadrature
    generate_quadrature_rule,
    quadrature_rule,
# Particles
    generate_particles,
    QuadraturePoints,
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
# transfer
    @P2G,
    @G2P,
    @G2P2G,
    @explain,
    ExplainedCode,
# implicit
    DofMap,
    BlockDofMap,
    dofmap,
    ndofs,
    dofs,
    create_sparse_matrix,
    create_block_sparse_matrix,
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

include("Core/Core.jl")
include("progress.jl")
import .Progress: @showprogress

include("NURBS/NURBS.jl")
include("Mesh/Mesh.jl")
include("Threading/Threading.jl")
include("Grids/Grids.jl")
include("particles.jl")
include("Basis/Basis.jl")
include("Transfer/Transfer.jl")
include("Implicit/Implicit.jl")
include("IO/IO.jl")

include("adapt.jl")
include("deprecated.jl")

include("Stencil/Stencil.jl")

end # module Tesserae
