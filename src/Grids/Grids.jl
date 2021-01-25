module Grids

using Reexport
using Coordinates
@reexport using Jams.TensorValues

using Base: @_inline_meta, @_propagate_inbounds_meta
using Base.Cartesian: @ntuple, @nall

export
# AbstractGrid
    AbstractGrid,
    gridaxes,
    gridorigin,
    gridsteps,
# Grid
    Grid,
# neighboring
    neighboring_nodes,
    neighboring_cells,
    whichcell

include("AbstractGrid.jl")
include("Grid.jl")
include("neighboring.jl")

end # module
