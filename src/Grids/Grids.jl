module Grids

using Reexport
using AxisArrays
@reexport using Jams.TensorValues

using Base: @_inline_meta, @_propagate_inbounds_meta
using Base.Cartesian: @ntuple, @nall

export
# GridBoundSet
    GridBoundSet,
# Grids
    AbstractGrid,
    CartesianGrid,
    Grid,
    gridaxes,
    gridorigin,
    gridsteps,
    getboundsets,
    getboundset,
    setboundset!,
# neighboring
    neighboring_nodes,
    neighboring_cells,
    whichcell

include("grid.jl")
include("neighboring.jl")

end # module
