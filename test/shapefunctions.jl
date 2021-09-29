@testset "BSplineValues" begin
    for T in (Float32, Float64)
        for dim in 1:2
            grid = Grid(ntuple(i -> 0.0:0.1:5.0, Val(dim)))
            for bspline in (LinearBSpline{dim}(), QuadraticBSpline{dim}(), CubicBSpline{dim}(),)
                it = Poingr.ShapeValues(T, bspline)
                for x in Iterators.product(ntuple(i -> 0.0:0.05:5.0, Val(dim))...)
                    update!(it, grid, Vec(x))
                    @test sum(it.N) ≈ 1
                    if !isa(bspline, LinearBSpline)
                        @test sum(it.∇N) ≈ zero(Vec{dim}) atol = sqrt(eps(T))
                    end
                end
            end
        end
    end
end

@testset "WLSValues" begin
    for T in (Float32, Float64)
        for dim in 1:2
            grid = Grid(ntuple(i -> 0.0:0.1:5.0, Val(dim)))
            for bspline in (QuadraticBSpline{dim}(), CubicBSpline{dim}(),)
                it = Poingr.ShapeValues(T, LinearWLS(bspline))
                for x in Iterators.product(ntuple(i -> 0.0:0.05:5.0, Val(dim))...)
                    update!(it, grid, Vec(x))
                    @test sum(it.N) ≈ 1
                    @test sum(it.∇N) ≈ zero(Vec{dim}) atol = sqrt(eps(T))
                end
            end
        end
    end
end
