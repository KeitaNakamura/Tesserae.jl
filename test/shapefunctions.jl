@testset "BSplineValues" begin
    for T in (Float32, Float64)
        for dim in 1:2
            grid = Grid(ntuple(i -> 0.0:0.1:5.0, Val(dim)))
            for bspline in (LinearBSpline(), QuadraticBSpline(), CubicBSpline(),)
                it = Poingr.ShapeValues{dim, T}(bspline)
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
            for bspline in (QuadraticBSpline(), CubicBSpline(),)
                it = Poingr.ShapeValues{dim, T}(LinearWLS(bspline))
                for x in Iterators.product(ntuple(i -> 0.0:0.05:5.0, Val(dim))...)
                    update!(it, grid, Vec(x))
                    @test sum(it.N) ≈ 1
                    @test sum(it.∇N) ≈ zero(Vec{dim}) atol = sqrt(eps(T))
                end
            end
        end
    end
end

@testset "GIMPValues" begin
    for T in (Float32, Float64)
        for dim in 1:2
            grid = Grid(ntuple(i -> 0.0:0.1:5.0, Val(dim)))
            for gimp in (GIMP(),)
                it = Poingr.ShapeValues{dim, T}(gimp)
                # GIMP doesn't have pertition of unity when closed to boundaries
                # if we follow eq.40 in Bardenhagen (2004)
                for x in Iterators.product(ntuple(i -> 1.0:0.05:4.0, Val(dim))...)
                    side_length = gridsteps(grid) ./ 2
                    r = Vec(side_length ./ 2)
                    update!(it, grid, Vec(x), r)
                    @test sum(it.N) ≈ 1
                    @test sum(it.∇N) ≈ zero(Vec{dim}) atol = sqrt(eps(T))
                end
            end
        end
    end
end
