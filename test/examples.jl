module TestCollision
    const RUN_TESTS = true
    cd(tempdir()) do
        include(joinpath(@__DIR__, "../docs/literate/examples/collision.jl"))
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

module TestDamBreak
    const RUN_TESTS = true
    cd(tempdir()) do
        include(joinpath(@__DIR__, "../docs/literate/examples/dam_break.jl"))
    end
end

module TestRigidBodyContact
    const RUN_TESTS = true
    cd(tempdir()) do
        include(joinpath(@__DIR__, "../docs/literate/examples/rigid_body_contact.jl"))
    end
end
