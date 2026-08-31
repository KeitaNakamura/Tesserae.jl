module Stencil

using StaticArrays
using StructArrays
using Tensorial

using KernelAbstractions
using Adapt

export
    Cell,
    Face,
    StencilArray,
    inner,
    foldpad!,
    mirrorpad!,
    stencil,
    stencil!,
    Diff,
    Gradient,
    Divergence,
    Laplacian,
    Curl

include("array.jl")
include("core.jl")
include("diff.jl")
include("tesserae.jl")

end # Stencil
