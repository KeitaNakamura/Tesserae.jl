module RigidBodies

using Jams.TensorValues

export
    RigidBody,
    center,
# Line
    Line,
    distance,
    normalunit,
    Polygon,
    Rectangle,
    isinside

include("RigidBody.jl")
include("Line.jl")
include("Polygon.jl")

end
