@testset "Interpolations" begin

@testset "Kernel" begin
@testset "BSpline" begin
    @testset "Fast calculations" begin
        getvals(itp, lattice, x) = [Marble.value(itp, lattice, I, x) for I in first(neighbornodes(itp, lattice, x))] |> vec
        getgrads(itp, lattice, x) = [gradient(x->Marble.value(itp, lattice, I, x), x) for I in first(neighbornodes(itp, lattice, x))] |> vec
        @testset "$itp" for (len,itp) in ((2,LinearBSpline()), (3,QuadraticBSpline()), (4,CubicBSpline()))
            @testset "dim=$dim" for dim in 1:3
                for T in (Float32, Float64)
                    lattice = Lattice(T, 1, ntuple(i->(-10,10), Val(dim))...)
                    # wrap by KernelCorrection because `BSpline` uses only fast version
                    mp = values(MPValues{dim, T}(KernelCorrection(itp), 1), 1)
                    x = rand(Vec{dim, T})
                    # fast version
                    N = Array{T}(undef, fill(len, dim)...)
                    ∇N = Array{Vec{dim, T}}(undef, fill(len, dim)...)
                    Marble.values_gradients!(N, reinterpret(reshape, T, ∇N), itp, lattice, x)
                    @test vec(N) ≈ getvals(itp, lattice, x)
                    @test vec(∇N) ≈ getgrads(itp, lattice, x)
                end
            end
        end
    end
    @testset "Consistency between `values_gradients` and `neighbornodes`" begin
        # check consistency between `values_gradients` and `neighbornodes` when a particle is exactly on the knot
        getvalues(itp, lattice, pt) = [Marble.value(itp, lattice, I, pt) for I in first(neighbornodes(itp, lattice, pt))]
        lattice = Lattice(1, (0,5))
        ## LinearBSpline
        itp = LinearBSpline()
        N = Marble.values_gradients(itp, 1.0) |> first |> Tuple |> collect
        @test N == [1, 0]
        @test N == getvalues(itp, lattice, Vec(2.0))
        ## QuadraticBSpline
        itp = QuadraticBSpline()
        N = Marble.values_gradients(itp, 1.5) |> first |> Tuple |> collect
        @test N == [0.5, 0.5, 0.0]
        @test N == getvalues(itp, lattice, Vec(2.5))
        ## CubicBSpline
        itp = CubicBSpline()
        N = Marble.values_gradients(itp, 2.0) |> first |> Tuple |> collect
        @test N ≈ [1/6, 4/6, 1/6, 0.0]
        @test N ≈ getvalues(itp, lattice, Vec(3.0))
    end
end
end # Kernel

@testset "MPValues" begin
@testset "BSpline" begin
    for T in (Float32, Float64)
        Random.seed!(1234)
        TOL = sqrt(eps(T))
        for dim in 1:3
            lattice = Lattice(T, 0.1, ntuple(i->(0,1), Val(dim))...)
            for itp in (LinearBSpline(), QuadraticBSpline(), CubicBSpline(),)
                mp = values(MPValues{dim, T}(itp, 1), 1)
                for _ in 1:100
                    x = rand(Vec{dim, T})
                    _, isfullyinside = neighbornodes(itp, lattice, x)
                    indices = update!(mp, itp, lattice, x)
                    CI = CartesianIndices(indices)
                    @test sum(mp.N[CI]) ≈ 1
                    @test sum(mp.∇N[CI]) ≈ zero(Vec{dim}) atol=TOL
                    if isfullyinside
                        @test mapreduce((j,i) -> mp.N[j]*lattice[i], +, CI, indices) ≈ x atol=TOL
                        @test mapreduce((j,i) -> lattice[i]⊗mp.∇N[j], +, CI, indices) ≈ I atol=TOL
                    end
                end
            end
        end
    end
end

@testset "WLS" begin
    for T in (Float32, Float64)
        Random.seed!(1234)
        TOL = sqrt(eps(T))
        for dim in 1:3
            lattice = Lattice(T, 0.1, ntuple(i->(0,1), Val(dim))...)
            l = spacing(lattice) / 2
            for kernel in (QuadraticBSpline(), CubicBSpline(), uGIMP())
                for WLS in (LinearWLS, Marble.BilinearWLS)
                    WLS == Marble.BilinearWLS && dim != 2 && continue
                    itp = WLS(kernel)
                    mp = values(MPValues{dim, T}(itp, 1), 1)
                    for _ in 1:100
                        x = rand(Vec{dim, T})
                        if kernel isa uGIMP
                            pt = (;x,l)
                        else
                            pt = x
                        end
                        indices = update!(mp, itp, lattice, pt)
                        CI = CartesianIndices(indices)
                        @test sum(mp.N[CI]) ≈ 1
                        @test sum(mp.∇N[CI]) ≈ zero(Vec{dim}) atol=TOL
                        @test mapreduce((j,i) -> mp.N[j]*lattice[i], +, CI, indices) ≈ x atol=TOL
                        @test mapreduce((j,i) -> lattice[i]⊗mp.∇N[j], +, CI, indices) ≈ I atol=TOL
                    end
                end
            end
        end
    end
end

@testset "uGIMP" begin
    for T in (Float32, Float64)
        Random.seed!(1234)
        TOL = sqrt(eps(T))
        for dim in 1:3
            lattice = Lattice(T, 0.1, ntuple(i->(0,1), Val(dim))...)
            for itp in (uGIMP(),)
                mp = values(MPValues{dim, T}(itp, 1), 1)
                l = spacing(lattice) / 2
                # uGIMP doesn't have pertition of unity when closed to boundaries
                # if we follow eq.40 in Bardenhagen (2004)
                for _ in 1:100
                    x = rand(Vec{dim, T})
                    pt = (;x,l)
                    _, isfullyinside = neighbornodes(itp, lattice, pt)
                    if isfullyinside
                        indices = update!(mp, itp, lattice, pt)
                        CI = CartesianIndices(indices)
                        @test sum(mp.N[CI]) ≈ 1
                        @test sum(mp.∇N[CI]) ≈ zero(Vec{dim}) atol=TOL
                        @test mapreduce((j,i) -> mp.N[j]*lattice[i], +, CI, indices) ≈ x atol=TOL
                        @test mapreduce((j,i) -> lattice[i]⊗mp.∇N[j], +, CI, indices) ≈ I atol=TOL
                    end
                end
            end
        end
    end
end

@testset "KernelCorrectionValue" begin
    @testset "$kernel" for kernel in (QuadraticBSpline(), CubicBSpline(), uGIMP())
        itp = KernelCorrection(kernel)
        for T in (Float32, Float64)
            Random.seed!(1234)
            TOL = sqrt(eps(T))
            for dim in 1:3
                lattice = Lattice(T, 0.1, ntuple(i->(0,1), Val(dim))...)
                l = spacing(lattice) / 2
                mp = values(MPValues{dim, T}(itp, 1), 1)
                for _ in 1:100
                    x = rand(Vec{dim, T})
                    if kernel isa uGIMP
                        pt = (;x,l)
                    else
                        pt = x
                    end
                    indices = update!(mp, itp, lattice, pt)
                    CI = CartesianIndices(indices)
                    @test sum(mp.N[CI]) ≈ 1
                    @test sum(mp.∇N[CI]) ≈ zero(Vec{dim}) atol=TOL
                    @test mapreduce((j,i) -> mp.N[j]*lattice[i], +, CI, indices) ≈ x atol=TOL
                    @test mapreduce((j,i) -> lattice[i]⊗mp.∇N[j], +, CI, indices) ≈ I atol=TOL
                end
            end
        end
    end
end

@testset "LinearWLS/KernelCorrection with sparsity pattern" begin
    for kernel in (QuadraticBSpline(), CubicBSpline())
        for Modifier in (LinearWLS, KernelCorrection)
            itp = Modifier(kernel)
            mp1 = values(MPValues{2}(itp, 1), 1)
            mp2 = values(MPValues{2}(itp, 1), 1)
            lattice = Lattice(1, (0,10), (0,10))
            sppat = trues(size(lattice))
            sppat[1:2, :] .= false
            sppat[:, 1:2] .= false
            xp1 = Vec(0.12,0.13)
            xp2 = xp1 .+ 2
            update!(mp1, itp, lattice, xp1)
            update!(mp2, itp, lattice, sppat, xp2)
            @test mp1.N[1:2, 1:2] ≈ mp2.N[2:3, 2:3]
            @test mp1.∇N[1:2, 1:2] ≈ mp2.∇N[2:3, 2:3]
        end
    end
end
end # MPValues

end # Interpolations
