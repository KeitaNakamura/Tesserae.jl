@testset "DruckerPrager" begin
    for elastic in (LinearElastic(E = 1e6, ν = 0.3),
                    SoilElastic(κ = 0.019, α = 40.0, p_ref = -1.0, μ_ref = 10.0),)
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
end
