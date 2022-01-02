@testset "MaterialModel constructors" begin
    # LinearElastic
    @test typeof(@inferred LinearElastic(; E = 1, K = 1)) <: LinearElastic{Float64}
    @test typeof(@inferred LinearElastic{Float32}(; E = 1, K = 1)) <: LinearElastic{Float32}
    # SoilHypoelastic
    @test typeof(@inferred SoilHypoelastic(; κ = 1, ν = 1, e0 = 1)) <: SoilHypoelastic{Float64}
    @test typeof(@inferred SoilHypoelastic{Float32}(; κ = 1, ν = 1, e0 = 1)) <: SoilHypoelastic{Float32}
    # SoilHyperelastic
    @test typeof(@inferred SoilHyperelastic(; κ = 1, α = 1, p_ref = 1, μ_ref = 1)) <: SoilHyperelastic{Float64}
    @test typeof(@inferred SoilHyperelastic{Float32}(; κ = 1, α = 1, p_ref = 1, μ_ref = 1)) <: SoilHyperelastic{Float32}
    # VonMises
    elastic = LinearElastic(; E = 1, K = 1)
    @test typeof(@inferred VonMises(elastic, q_y = 1)) <: VonMises{Float64}
    @test typeof(@inferred VonMises(elastic, :plane_strain; c = 1)) <: VonMises{Float64}
    @test typeof(@inferred VonMises{Float32}(elastic, q_y = 1)) <: VonMises{Float32}
    @test typeof(@inferred VonMises{Float32}(elastic, :plane_strain; c = 1)) <: VonMises{Float32}
    # DruckerPrager
    for elastic in (LinearElastic(; E = 1, K = 1),
                    SoilHypoelastic(; κ = 1, ν = 1, e0 = 1),
                    SoilHyperelastic(; κ = 1, α = 1, p_ref = 1, μ_ref = 1))
        @test typeof(@inferred DruckerPrager(elastic, :plane_strain; c = 1, ϕ = 1)) <: DruckerPrager{Float64}
        @test typeof(@inferred DruckerPrager(elastic, :circumscribed; c = 1, ϕ = 1)) <: DruckerPrager{Float64}
        @test typeof(@inferred DruckerPrager(elastic, :inscribed; c = 1, ϕ = 1)) <: DruckerPrager{Float64}
        @test typeof(@inferred DruckerPrager{Float32}(elastic, :plane_strain; c = 1, ϕ = 1)) <: DruckerPrager{Float32}
        @test typeof(@inferred DruckerPrager{Float32}(elastic, :circumscribed; c = 1, ϕ = 1)) <: DruckerPrager{Float32}
        @test typeof(@inferred DruckerPrager{Float32}(elastic, :inscribed; c = 1, ϕ = 1)) <: DruckerPrager{Float32}
        @test DruckerPrager(elastic, :inscribed; c = 1, ϕ = 1, tension_cutoff = false).tension_cutoff == Inf
    end
    # NewtonianFluid
    @test typeof(@inferred NewtonianFluid(; ρ0 = 1, P0 = 1, c = 1, μ = 1)) <: NewtonianFluid{Float64, Poingr.SimpleWaterModel{Float64}}
    @test typeof(@inferred NewtonianFluid{Float32}(; ρ0 = 1, P0 = 1, c = 1, μ = 1)) <: NewtonianFluid{Float32, Poingr.SimpleWaterModel{Float32}}
end
