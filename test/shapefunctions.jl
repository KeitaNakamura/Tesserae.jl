@testset "BSplineValues" begin
    for dim in 1:2
        grid = Grid(ntuple(i -> 0.0:0.1:5.0, Val(dim)))
        for bspline in (LinearBSpline{dim}(), QuadraticBSpline{dim}(), CubicBSpline{dim}(),)
            it = Poingr.ShapeValues(bspline)
            for x in Iterators.product(ntuple(i -> 0.0:0.05:5.0, Val(dim))...)
                update!(it, grid, Vec(x))
                @test sum(it.N) ≈ 1
                if !isa(bspline, LinearBSpline)
                    @test sum(it.∇N) ≈ zero(Vec{dim}) atol = sqrt(eps(Float64))
                end
            end
        end
    end
end

@testset "WLSValues" begin
    for dim in 1:2
        grid = Grid(ntuple(i -> 0.0:0.1:5.0, Val(dim)))
        for bspline in (QuadraticBSpline{dim}(), CubicBSpline{dim}(),)
            it = Poingr.ShapeValues(WLS{1}(bspline))
            for x in Iterators.product(ntuple(i -> 0.0:0.05:5.0, Val(dim))...)
                update!(it, grid, Vec(x))
                @test sum(it.N) ≈ 1
                @test sum(it.∇N) ≈ zero(Vec{dim}) atol = sqrt(eps(Float64))
            end
        end
    end
end
