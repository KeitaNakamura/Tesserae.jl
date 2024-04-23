module TestElasticImpact
    const RUN_TESTS = true
    cd(tempdir()) do
        include(joinpath(@__DIR__, "../docs/literate/examples/elastic_impact.jl"))
    end
end

module TestTLMPMVortex
    const RUN_TESTS = true
    cd(tempdir()) do
        include(joinpath(@__DIR__, "../docs/literate/examples/tlmpm_vortex.jl"))
    end
end

module TestImplicitJacobianFree
    const RUN_TESTS = true
    cd(tempdir()) do
        include(joinpath(@__DIR__, "../docs/literate/examples/implicit_jacobian_free.jl"))
    end
end

module TestImplicitJacobianBased
    const RUN_TESTS = true
    cd(tempdir()) do
        include(joinpath(@__DIR__, "../docs/literate/examples/implicit_jacobian_based.jl"))
    end
end
