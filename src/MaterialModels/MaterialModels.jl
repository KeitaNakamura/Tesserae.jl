module MaterialModels

using Base: @_inline_meta

using Reexport
@reexport using Jams.TensorValues

export
    MaterialModel,
    LinearElastic,
    SoilElastic,
    VonMises,
    DruckerPrager,
    WaterModel,
    update_stress,
    soundspeed,
    volumetric_stress,
    deviatoric_stress,
    volumetric_strain,
    deviatoric_strain,
    infinitesimal_strain

abstract type MaterialModel end

include("utils.jl")
include("LinearElastic.jl")
include("SoilElastic.jl")
include("VonMises.jl")
include("DruckerPrager.jl")
include("WaterModel.jl")

end
