module Poingr

include("TensorValues/TensorValues.jl")
include("DofHelpers/DofHelpers.jl")
include("Arrays/Arrays.jl")
include("MaterialModels/MaterialModels.jl")
include("Collections/Collections.jl")
include("Grids/Grids.jl")
include("ShapeFunctions/ShapeFunctions.jl")
include("States/States.jl")
include("MPSpaces/MPSpaces.jl")
include("ContactMechanics/ContactMechanics.jl")
include("Loggers/Loggers.jl")
include("VTKOutputs/VTKOutputs.jl")

# Exports
using Reexport
@reexport using Poingr.TensorValues
@reexport using Poingr.DofHelpers
@reexport using Poingr.Arrays
@reexport using Poingr.MaterialModels
@reexport using Poingr.Collections
@reexport using Poingr.Grids
@reexport using Poingr.ShapeFunctions
@reexport using Poingr.States
@reexport using Poingr.MPSpaces
@reexport using Poingr.ContactMechanics
@reexport using Poingr.Loggers
@reexport using Poingr.VTKOutputs

end # module
