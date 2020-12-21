module Jams

include("Grids/Grids.jl")
include("Interpolations/Interpolations.jl")
include("DofHelpers/DofHelpers.jl")
include("Arrays/Arrays.jl")

# Exports
using Reexport
@reexport using Jams.Grids
@reexport using Jams.Interpolations
@reexport using Jams.DofHelpers
@reexport using Jams.Arrays

end # module
