@testset "Interpolations" begin

@testset "BSplineValue" begin
    for T in (Float32, Float64)
        Random.seed!(1234)
        TOL = sqrt(eps(T))
        for dim in 1:3
            lattice = Lattice(0.1, ntuple(i->(0,1), Val(dim))...)
            for bspline in (LinearBSpline(), QuadraticBSpline(), CubicBSpline(),)
                mp = MPValue{dim, T}(bspline)
                for _ in 1:100
                    x = rand(Vec{dim, T})
                    inds = neighbornodes(mp, lattice, x)
                    update!(mp, lattice, inds, x)
                    @test sum(mp.N) ≈ 1
                    @test sum(mp.∇N) ≈ zero(Vec{dim}) atol=TOL
                    l = Marble.get_supportlength(bspline)
                    if all(a->l<a<1-l, x)
                        @test mapreduce((N,∇N,i) -> N*lattice[i], +, mp.N, mp.∇N, inds) ≈ x atol=TOL
                        @test mapreduce((N,∇N,i) -> N*lattice[i], +, mp.N, mp.∇N, inds) ≈ x atol=TOL
                        @test mapreduce((N,∇N,i) -> lattice[i]⊗∇N, +, mp.N, mp.∇N, inds) ≈ I atol=TOL
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
            r = spacing(lattice) / 4
            for kernel in (QuadraticBSpline(), CubicBSpline(), GIMP())
                for WLS in (LinearWLS, Marble.BilinearWLS)
                    WLS == Marble.BilinearWLS && dim != 2 && continue
                    mp = MPValue{dim, T}(WLS(kernel))
                    for _ in 1:100
                        x = rand(Vec{dim, T})
                        if kernel isa GIMP
                            pt = (;x,r)
                        else
                            pt = x
                        end
                        inds = neighbornodes(mp, lattice, pt)
                        update!(mp, lattice, inds, pt)
                        @test sum(mp.N) ≈ 1
                        @test sum(mp.∇N) ≈ zero(Vec{dim}) atol=TOL
                        @test mapreduce((N,∇N,i) -> N*lattice[i], +, mp.N, mp.∇N, inds) ≈ x atol=TOL
                        @test mapreduce((N,∇N,i) -> lattice[i]⊗∇N, +, mp.N, mp.∇N, inds) ≈ I atol=TOL
                    end
                end
            end
        end
    end
end

@testset "GIMPValue" begin
    for T in (Float32, Float64)
        Random.seed!(1234)
        TOL = sqrt(eps(T))
        for dim in 1:3
            lattice = Lattice(0.1, ntuple(i->(0,1), Val(dim))...)
            for gimp in (GIMP(),)
                mp = MPValue{dim, T}(gimp)
                r = spacing(lattice) / 4
                # GIMP doesn't have pertition of unity when closed to boundaries
                # if we follow eq.40 in Bardenhagen (2004)
                for _ in 1:100
                    x = rand(Vec{dim, T})
                    if all(r .< x .< 1-r)
                        pt = (;x,r)
                        inds = neighbornodes(mp, lattice, pt)
                        update!(mp, lattice, inds, pt)
                        @test sum(mp.N) ≈ 1
                        @test sum(mp.∇N) ≈ zero(Vec{dim}) atol=TOL
                        @test mapreduce((N,∇N,i) -> N*lattice[i], +, mp.N, mp.∇N, inds) ≈ x atol=TOL
                        @test mapreduce((N,∇N,i) -> lattice[i]⊗∇N, +, mp.N, mp.∇N, inds) ≈ I atol=TOL
                    end
                end
            end
        end
    end
end

@testset "KernelCorrectionValue" begin
    for T in (Float32, Float64)
        Random.seed!(1234)
        TOL = sqrt(eps(T))
        for dim in 1:3
            lattice = Lattice(0.1, ntuple(i->(0,1), Val(dim))...)
            r = spacing(lattice) / 4
            for kernel in (QuadraticBSpline(), CubicBSpline(), GIMP())
                mp = MPValue{dim, T}(KernelCorrection(kernel))
                for _ in 1:100
                    x = rand(Vec{dim, T})
                    if kernel isa GIMP
                        pt = (;x,r)
                    else
                        pt = x
                    end
                    inds = neighbornodes(mp, lattice, pt)
                    update!(mp, lattice, inds, pt)
                    @test sum(mp.N) ≈ 1
                    @test sum(mp.∇N) ≈ zero(Vec{dim}) atol=TOL
                    @test mapreduce((N,∇N,i) -> N*lattice[i], +, mp.N, mp.∇N, inds) ≈ x atol=TOL
                    @test mapreduce((N,∇N,i) -> lattice[i]⊗∇N, +, mp.N, mp.∇N, inds) ≈ I atol=TOL
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
            inds1 = neighbornodes(mp1, lattice, xp1)
            inds2 = neighbornodes(mp1, lattice, xp2)
            update!(mp1, lattice, inds1, xp1)
            update!(mp2, lattice, sppat, inds2, xp2)
            @test mp1.N ≈ filter(!iszero, mp2.N)
            @test mp1.∇N ≈ filter(!iszero, mp2.∇N)
        end
    end
end

end # Interpolations
