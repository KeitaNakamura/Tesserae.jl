module TestCollision
    const RUN_TESTS = true
    cd(tempdir()) do
        include(joinpath(@__DIR__, "../docs/literate/tutorials/collision.jl"))
    end
end

module TestCPDI
    const RUN_TESTS = true
    cd(tempdir()) do
        include(joinpath(@__DIR__, "../docs/literate/tutorials/cpdi.jl"))
    end
end

module TestTLMPMVortex
    const RUN_TESTS = true
    cd(tempdir()) do
        include(joinpath(@__DIR__, "../docs/literate/tutorials/tlmpm_vortex.jl"))
    end
end

module TestImplicitJacobianFree
    const RUN_TESTS = true
    cd(tempdir()) do
        include(joinpath(@__DIR__, "../docs/literate/tutorials/implicit_jacobian_free.jl"))
    end
end

module TestImplicitJacobianBased
    const RUN_TESTS = true
    cd(tempdir()) do
        include(joinpath(@__DIR__, "../docs/literate/tutorials/implicit_jacobian_based.jl"))
    end
end

module TestCollapse
    const RUN_TESTS = true
    cd(tempdir()) do
        include(joinpath(@__DIR__, "../docs/literate/tutorials/collapse.jl"))
    end
end

module TestDamBreak
    const RUN_TESTS = true
    cd(tempdir()) do
        include(joinpath(@__DIR__, "../docs/literate/tutorials/dam_break.jl"))
    end
end

module TestRigidBodyContact
    const RUN_TESTS = true
    cd(tempdir()) do
        include(joinpath(@__DIR__, "../docs/literate/tutorials/rigid_body_contact.jl"))
    end
end

module TestTaylorImpact
    const RUN_TESTS = true
    cd(tempdir()) do
        include(joinpath(@__DIR__, "../docs/literate/tutorials/taylor_impact.jl"))
    end
end

module TestHeat
    const RUN_TESTS = true
    cd(tempdir()) do
        include(joinpath(@__DIR__, "../docs/literate/tutorials/heat.jl"))
    end
end
