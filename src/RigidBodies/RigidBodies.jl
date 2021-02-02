module RigidBodies

using Jams.TensorValues

export
    RigidBody,
    center,
    translate!,
# Line
    Line,
    distance,
    normalunit,
# Polygon
    Polygon,
    Rectangle,
    isinside

include("RigidBody.jl")
include("Line.jl")
include("Polygon.jl")

end
