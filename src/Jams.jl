module Jams

include("TensorValues/TensorValues.jl")
include("DofHelpers/DofHelpers.jl")
include("Arrays/Arrays.jl")
include("Collections/Collections.jl")
include("MaterialModels/MaterialModels.jl")
include("Grids/Grids.jl")
include("ShapeFunctions/ShapeFunctions.jl")
include("States/States.jl")
include("MPSpaces/MPSpaces.jl")
include("ContactMechanics/ContactMechanics.jl")
include("RigidBodies/RigidBodies.jl")
include("VTKOutputs/VTKOutputs.jl")

# Exports
using Reexport
@reexport using Jams.TensorValues
@reexport using Jams.DofHelpers
@reexport using Jams.Arrays
@reexport using Jams.Collections
@reexport using Jams.MaterialModels
@reexport using Jams.Grids
@reexport using Jams.ShapeFunctions
@reexport using Jams.States
@reexport using Jams.MPSpaces
@reexport using Jams.ContactMechanics
@reexport using Jams.RigidBodies
@reexport using Jams.VTKOutputs

end # module
