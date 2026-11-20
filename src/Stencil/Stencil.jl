module Stencil

import ..Tesserae: boundaries, generate_grid, spacing, spacing_inv, volume, create_sparse_matrix
using ..Tesserae: AbstractMesh, AbstractDevice, CartesianMesh, Vec, Grid, StructArray, check_gridproperty, get_mesh, block_size_log2, @_propagate_inbounds_meta
using ..Tesserae: append_dofs!, node_dofs, check_matrix_entry_size, matrix_parent, matrix_parent_indices, mapped_index
using SparseArrays: SparseMatrixCSC, sparse, rowvals, nonzeros, nzrange
import Adapt, KernelAbstractions
using OffsetArrays: OffsetArray, no_offset_view

export
    Cell,
    Face,
    Edge,
    Vertex,
    Region,
    interior,
    boundaries,
    unitoffsets,
    coordinates,
    AxisPlane,
    reflect,
    halos,
    regionindices,
    @stencil,
    @stencil_assemble

include("offset.jl")
include("frame.jl")
include("region.jl")
include("geometry.jl")
include("indices.jl")
include("coordinates.jl")
include("broadcast.jl")
include("macro.jl")
include("matrix.jl")
include("grid.jl")

end # Stencil
