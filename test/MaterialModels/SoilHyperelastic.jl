@testset "SoilHyperelastic" begin
    model = SoilHyperelastic(κ = 0.01, α = 40.0, p_ref = -98.0, μ_ref = 6000.0)
    ϵᵉ = -rand(SymmetricSecondOrderTensor{3}) * 0.01
    σ = (@inferred matcalc(Val(:stress), model, ϵᵉ))
    @test (@inferred matcalc(Val(:elastic_strain), model, σ)) ≈ ϵᵉ
    @test (@inferred Poingr.W(model, ϵᵉ)) + (@inferred Poingr.W̃(model, σ)) ≈ σ ⊡ ϵᵉ
    @test (@inferred Poingr.∇W(model, ϵᵉ)) ≈ gradient(ϵᵉ -> Poingr.W(model, ϵᵉ), ϵᵉ)
    @test (@inferred Poingr.∇²W(model, ϵᵉ)) ≈ hessian(ϵᵉ -> Poingr.W(model, ϵᵉ), ϵᵉ)
    @test (@inferred Poingr.∇W̃(model, σ)) ≈ gradient(σ -> Poingr.W̃(model, σ), σ)
    @test (@inferred Poingr.∇²W̃(model, σ)) ≈ hessian(σ -> Poingr.W̃(model, σ), σ)
end
