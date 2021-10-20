@testset "DruckerPrager" begin
    for elastic in (LinearElastic(E = 1e6, ν = 0.3),
                    SoilHyperelastic(κ = 0.019, α = 40.0, p_ref = -1.0, μ_ref = 10.0),)
        for mc_type in (:circumscribed, :inscribed, :plane_strain)
            Random.seed!(1234)
            steps = 100
            model = @inferred DruckerPrager(elastic, mc_type, c = 1e3, ϕ = 30, ψ = 10)
            σ = -100.0*one(SymmetricSecondOrderTensor{3})
            dϵ = -0.1*rand(SymmetricSecondOrderTensor{3}) / steps
            for i in 1:steps
                σ = @inferred update_stress(model, σ, dϵ)
                f = @inferred Poingr.yield_function(model, σ)
                if f > 0
                    @test f < 1e-5
                end
            end
        end
    end
    @testset "tension cut-off" begin
        elastic = LinearElastic(E = 1e6, ν = 0.3)
        model = DruckerPrager(elastic, :circumscribed, c = 20.0, ϕ = 30, ψ = 0, tension_cutoff = 10.0)
        n = normalize(dev(rand(SymmetricSecondOrderTensor{3})))
        # 1
        σ = 10.0*I + 10.0*n
        @test Poingr.tension_cutoff(model, σ) ≈ σ
        @test Poingr.yield_function(model, Poingr.tension_cutoff(model, σ)) < 0
        # 2
        σ = 20.0*I + 10.0*n
        @test Poingr.tension_cutoff(model, σ) ≈ 10.0*I + 10.0*n
        @test Poingr.yield_function(model, Poingr.tension_cutoff(model, σ)) < 0
        # 3
        σ = 20.0*I + 30.0*n
        @test mean(Poingr.tension_cutoff(model, σ)) ≈ 10.0
        @test abs(Poingr.yield_function(model, Poingr.tension_cutoff(model, σ))) < sqrt(eps(Float64))
    end
end
