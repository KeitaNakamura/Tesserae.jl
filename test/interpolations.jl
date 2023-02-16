@testset "Interpolations" begin

@testset "BSplineValue" begin
    for T in (Float32, Float64)
        Random.seed!(1234)
        TOL = sqrt(eps(T))
        for dim in 1:3
            lattice = Lattice(0.1, ntuple(i->(0,1), Val(dim))...)
            for itp in (LinearBSpline(), QuadraticBSpline(), CubicBSpline(),)
                mp = MPValue{dim, T}(itp)
                for _ in 1:100
                    x = rand(Vec{dim, T})
                    _, isnearbounds = neighbornodes(itp, lattice, x)
                    indices = update!(mp, lattice, x)
                    @test sum(mp.N) ≈ 1
                    @test sum(mp.∇N) ≈ zero(Vec{dim}) atol=TOL
                    if !isnearbounds
                        @test mapreduce((N,∇N,i) -> N*lattice[i], +, mp.N, mp.∇N, indices) ≈ x atol=TOL
                        @test mapreduce((N,∇N,i) -> lattice[i]⊗∇N, +, mp.N, mp.∇N, indices) ≈ I atol=TOL
                    end
                end
            end
        end
    end
end

@testset "WLSValue" begin
    for T in (Float32, Float64)
        Random.seed!(1234)
        TOL = sqrt(eps(T))
        for dim in 1:3
            lattice = Lattice(0.1, ntuple(i->(0,1), Val(dim))...)
            l = spacing(lattice) / 2
            for kernel in (QuadraticBSpline(), CubicBSpline(), uGIMP())
                for WLS in (LinearWLS, Marble.BilinearWLS)
                    WLS == Marble.BilinearWLS && dim != 2 && continue
                    mp = MPValue{dim, T}(WLS(kernel))
                    for _ in 1:100
                        x = rand(Vec{dim, T})
                        if kernel isa uGIMP
                            pt = (;x,l)
                        else
                            pt = x
                        end
                        indices = update!(mp, lattice, pt)
                        @test sum(mp.N) ≈ 1
                        @test sum(mp.∇N) ≈ zero(Vec{dim}) atol=TOL
                        @test mapreduce((N,∇N,i) -> N*lattice[i], +, mp.N, mp.∇N, indices) ≈ x atol=TOL
                        @test mapreduce((N,∇N,i) -> lattice[i]⊗∇N, +, mp.N, mp.∇N, indices) ≈ I atol=TOL
                    end
                end
            end
        end
    end
end

@testset "uGIMPValue" begin
    for T in (Float32, Float64)
        Random.seed!(1234)
        TOL = sqrt(eps(T))
        for dim in 1:3
            lattice = Lattice(0.1, ntuple(i->(0,1), Val(dim))...)
            for itp in (uGIMP(),)
                mp = MPValue{dim, T}(itp)
                l = spacing(lattice) / 2
                # uGIMP doesn't have pertition of unity when closed to boundaries
                # if we follow eq.40 in Bardenhagen (2004)
                for _ in 1:100
                    x = rand(Vec{dim, T})
                    pt = (;x,l)
                    _, isnearbounds = neighbornodes(itp, lattice, pt)
                    if !isnearbounds
                        indices = update!(mp, lattice, pt)
                        @test sum(mp.N) ≈ 1
                        @test sum(mp.∇N) ≈ zero(Vec{dim}) atol=TOL
                        @test mapreduce((N,∇N,i) -> N*lattice[i], +, mp.N, mp.∇N, indices) ≈ x atol=TOL
                        @test mapreduce((N,∇N,i) -> lattice[i]⊗∇N, +, mp.N, mp.∇N, indices) ≈ I atol=TOL
                    end
                end
            end
        end
    end
end

@testset "KernelCorrectionValue" begin
    @testset "$kernel" for kernel in (QuadraticBSpline(), CubicBSpline(), uGIMP())
        for T in (Float32, Float64)
            Random.seed!(1234)
            TOL = sqrt(eps(T))
            for dim in 1:3
                lattice = Lattice(0.1, ntuple(i->(0,1), Val(dim))...)
                l = spacing(lattice) / 2
                mp = MPValue{dim, T}(KernelCorrection(kernel))
                for _ in 1:100
                    x = rand(Vec{dim, T})
                    if kernel isa uGIMP
                        pt = (;x,l)
                    else
                        pt = x
                    end
                    indices = update!(mp, lattice, pt)
                    @test sum(mp.N) ≈ 1
                    @test sum(mp.∇N) ≈ zero(Vec{dim}) atol=TOL
                    @test mapreduce((N,∇N,i) -> N*lattice[i], +, mp.N, mp.∇N, indices) ≈ x atol=TOL
                    @test mapreduce((N,∇N,i) -> lattice[i]⊗∇N, +, mp.N, mp.∇N, indices) ≈ I atol=TOL
                end
            end
        end
    end
end

@testset "LinearWLS/KernelCorrection with sparsity pattern" begin
    for kernel in (QuadraticBSpline(), CubicBSpline())
        for Modifier in (LinearWLS, KernelCorrection)
            mp1 = MPValue{2}(Modifier(kernel))
            mp2 = MPValue{2}(Modifier(kernel))
            lattice = Lattice(1, (0,10), (0,10))
            sppat = trues(size(lattice))
            sppat[1:2, :] .= false
            sppat[:, 1:2] .= false
            xp1 = Vec(0.12,0.13)
            xp2 = xp1 .+ 2
            update!(mp1, lattice, xp1)
            update!(mp2, lattice, sppat, xp2)
            @test mp1.N ≈ filter(!iszero, mp2.N)
            @test mp1.∇N ≈ filter(!iszero, mp2.∇N)
        end
    end
end

end # Interpolations
