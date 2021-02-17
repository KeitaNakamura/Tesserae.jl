@testset "SoilElastic" begin
    model = SoilElastic(κ = 0.01, α = 40.0, p_ref = -98.0, μ_ref = 6000.0)
    ϵᵉ = -rand(SymmetricSecondOrderTensor{3}) * 0.01
    σ = (@inferred MaterialModels.compute_stress(model, ϵᵉ))
    @test (@inferred MaterialModels.W(model, ϵᵉ)) + (@inferred MaterialModels.W̃(model, σ)) ≈ σ ⊡ ϵᵉ
    @test (@inferred MaterialModels.∇W(model, ϵᵉ)) ≈ gradient(ϵᵉ -> MaterialModels.W(model, ϵᵉ), ϵᵉ)
    @test (@inferred MaterialModels.∇²W(model, ϵᵉ)) ≈ hessian(ϵᵉ -> MaterialModels.W(model, ϵᵉ), ϵᵉ)
    @test (@inferred MaterialModels.compute_elastic_strain(model, σ)) ≈ ϵᵉ
    @test (@inferred MaterialModels.∇W̃(model, σ)) ≈ gradient(σ -> MaterialModels.W̃(model, σ), σ)
    @test (@inferred MaterialModels.∇²W̃(model, σ)) ≈ hessian(σ -> MaterialModels.W̃(model, σ), σ)
end
