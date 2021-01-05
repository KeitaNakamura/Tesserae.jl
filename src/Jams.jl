module Jams

include("TensorValues/TensorValues.jl")
include("DofHelpers/DofHelpers.jl")
include("Arrays/Arrays.jl")
include("Grids/Grids.jl")
include("PointStates/PointStates.jl")
include("ShapeFunctions/ShapeFunctions.jl")
include("MPSpaces/MPSpaces.jl")

# Exports
using Reexport
using Jams.TensorValues: ∇, ∇ₛ; export ∇, ∇ₛ
@reexport using Jams.DofHelpers
@reexport using Jams.Arrays
@reexport using Jams.Grids
@reexport using Jams.PointStates
@reexport using Jams.ShapeFunctions
@reexport using Jams.MPSpaces

end # module
