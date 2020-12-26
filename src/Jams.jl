module Jams

include("TensorValues/TensorValues.jl")
include("Arrays/Arrays.jl")
include("Grids/Grids.jl")
include("Interpolations/Interpolations.jl")
include("DofHelpers/DofHelpers.jl")

# Exports
using Reexport
using Jams.TensorValues: ∇, ∇ₛ; export ∇, ∇ₛ
@reexport using Jams.Grids
@reexport using Jams.Interpolations
@reexport using Jams.DofHelpers
@reexport using Jams.Arrays

end # module
