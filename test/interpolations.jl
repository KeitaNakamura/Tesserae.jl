@testset "MPValue" begin
    for dim in (1,2,3)
        @test eltype(MPValue(Vec{dim}, QuadraticBSpline())) ==
              eltype(MPValue(Vec{dim, Float64}, QuadraticBSpline()))
        for T in (Float32, Float64)
            for kernel in (LinearBSpline(), QuadraticBSpline(), CubicBSpline(), GIMP())
                for extension in (identity, WLS, KernelCorrection)
                    it = extension(kernel)
                    mp = @inferred MPValue(Vec{dim,T}, it)
                    @test interpolation(mp) === it
                    @test mp.N isa Array{T}
                    @test mp.∇N isa Array{Vec{dim,T}}
                    @test ndims(mp.N) == dim
                    @test ndims(mp.∇N) == dim
                    @test size(mp.N) == size(mp.∇N)
                    @test all(size(neighboringnodes(mp)) .< size(mp.N))
                end
            end
        end
    end
end

@testset "MPValueVector" begin
    for dim in (1,2,3)
        @test eltype(MPValueVector(Vec{dim}, QuadraticBSpline(), 2)) ==
              eltype(MPValueVector(Vec{dim, Float64}, QuadraticBSpline(), 2))
        for T in (Float32, Float64)
            n = 100
            mpvalues = @inferred MPValueVector(Vec{dim,T}, QuadraticBSpline(), n)
            @test size(mpvalues) === (n,)
            @test interpolation(mpvalues) === QuadraticBSpline()
            @test all(eachindex(mpvalues)) do i
                typeof(mpvalues[1]) === eltype(mpvalues)
            end
        end
    end
end

@testset "Interpolations" begin

    function check_partition_of_unity(mp, x; atol=sqrt(eps(eltype(mp.N))))
        indices = neighboringnodes(mp)
        CI = CartesianIndices(indices) # local indices
        isapprox(sum(mp.N[CI]), 1) && isapprox(sum(mp.∇N[CI]), zero(eltype(mp.∇N)); atol)
    end
    function check_linear_field_reproduction(mp, x, X; atol=sqrt(eps(eltype(mp.N))))
        indices = neighboringnodes(mp)
        CI = CartesianIndices(indices) # local indices
        isapprox(mapreduce((j,i) -> X[i]*mp.N[j],  +, CI, indices), x; atol) &&
        isapprox(mapreduce((j,i) -> X[i]⊗mp.∇N[j], +, CI, indices), I; atol)
    end

    @testset "$it" for it in (LinearBSpline(), QuadraticBSpline(), CubicBSpline())
        for T in (Float32, Float64), dim in (1,2,3), it in (LinearBSpline(), QuadraticBSpline(), CubicBSpline(),)
            Random.seed!(1234)
            mp = MPValue(Vec{dim,T}, it)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            @test all(1:100) do _
                x = rand(Vec{dim, T})
                update!(mp, x, mesh)
                isnearbounds = size(mp.N) != size(neighboringnodes(mp))
                PU = check_partition_of_unity(mp, x)
                LFR = check_linear_field_reproduction(mp, x, mesh)
                isnearbounds ? (PU && !LFR) : (PU && LFR)
            end
        end
    end

    @testset "GIMP()" begin
        it = GIMP()
        for T in (Float32, Float64), dim in (1,2,3)
            Random.seed!(1234)
            mp = MPValue(Vec{dim,T}, it)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            l = 0.5*spacing(mesh) / 2
            @test all(1:100) do _
                x = rand(Vec{dim, T})
                update!(mp, (;x,l), mesh)
                isnearbounds = any(.!(l .< x .< 1-l))
                PU = check_partition_of_unity(mp, x)
                LFR = check_linear_field_reproduction(mp, x, mesh)
                # GIMP doesn't have pertition of unity when very closed to boundaries
                # if we follow eq.40 in Bardenhagen (2004)
                isnearbounds ? (!PU && !LFR) : (PU && LFR) # GIMP
            end
        end
    end

    @testset "WLS($kernel)" for kernel in (LinearBSpline(), QuadraticBSpline(), CubicBSpline(), GIMP())
        it = WLS(kernel)
        for T in (Float32, Float64), dim in (1,2,3)
            Random.seed!(1234)
            mp = MPValue(Vec{dim,T}, it)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            l = 0.5*spacing(mesh) / 2
            @test all(1:100) do _
                x = rand(Vec{dim, T})
                update!(mp, (;x,l), mesh)
                PU = check_partition_of_unity(mp, x)
                LFR = check_linear_field_reproduction(mp, x, mesh)
                PU && LFR
            end
        end
    end

    @testset "KernelCorrection($kernel)" for kernel in (LinearBSpline(), QuadraticBSpline(), CubicBSpline(), GIMP())
        it = KernelCorrection(kernel)
        for T in (Float32, Float64), dim in (1,2,3)
            Random.seed!(1234)
            mp = MPValue(Vec{dim,T}, it)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            l = 0.5*spacing(mesh) / 2
            @test all(1:100) do _
                x = rand(Vec{dim, T})
                update!(mp, (;x,l), mesh)
                PU = check_partition_of_unity(mp, x)
                LFR = check_linear_field_reproduction(mp, x, mesh)
                PU && LFR
            end
        end
    end

end
