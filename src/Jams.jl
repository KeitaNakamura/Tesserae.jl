module Jams

include("TensorValues/TensorValues.jl")
include("DofHelpers/DofHelpers.jl")
include("MaterialModels/MaterialModels.jl")
include("Arrays/Arrays.jl")
include("Grids/Grids.jl")
include("States/States.jl")
include("ShapeFunctions/ShapeFunctions.jl")
include("MPSpaces/MPSpaces.jl")
include("VTKOutputs/VTKOutputs.jl")

# Exports
using Reexport
@reexport using Jams.TensorValues
@reexport using Jams.DofHelpers
@reexport using Jams.MaterialModels
@reexport using Jams.Arrays
@reexport using Jams.Grids
@reexport using Jams.States
@reexport using Jams.ShapeFunctions
@reexport using Jams.MPSpaces
@reexport using Jams.VTKOutputs

end # module
