module Grids

using Reexport
using AxisArrays
@reexport using Jams.TensorValues

using Base: @_inline_meta, @_propagate_inbounds_meta
using Base.Cartesian: @ntuple, @nall

export
# GridBoundSet
    GridBoundSet,
# AbstractGrid
    AbstractGrid,
    gridaxes,
    gridorigin,
    gridsteps,
    getboundsets,
    getboundset,
    setboundset!,
# CartesianGrid
    CartesianGrid,
# Grid
    Grid,
# neighboring
    neighboring_nodes,
    neighboring_cells,
    whichcell

include("GridBoundSet.jl")
include("AbstractGrid.jl")
include("CartesianGrid.jl")
include("Grid.jl")
include("neighboring.jl")

end # module
