const RUN_TESTS = true

@testset "Examples" begin
    @testset "Elastic impact between two rings" begin
        cd(tempdir()) do
            include(joinpath(@__DIR__, "../docs/literate/examples/elastic_impact.jl"))
        end
    end
end
