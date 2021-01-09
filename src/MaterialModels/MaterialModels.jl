module MaterialModels

using Base: @_inline_meta

using Reexport
@reexport using Jams.TensorValues

export
    MaterialModel,
    LinearElastic,
    VonMises,
    DruckerPrager,
    WaterModel,
    update_stress,
    soundspeed,
    infinitesimal_strain

abstract type MaterialModel end

include("utils.jl")
include("linearelastic.jl")
include("vonmises.jl")
include("druckerprager.jl")
include("water.jl")

end
