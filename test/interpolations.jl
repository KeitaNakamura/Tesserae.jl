@testset "BSplineValues" begin
    for T in (Float32, Float64)
        Random.seed!(1234)
        TOL = sqrt(eps(T))
        for dim in 1:3
            # using T=Float32 in Grid fails
            grid = Grid(ntuple(i -> 0:0.1:1, Val(dim)))
            for bspline in (LinearBSpline(), QuadraticBSpline(), CubicBSpline(),)
                mpvalues = MPValues{dim, T}(bspline)
                for _ in 1:2000
                    x = rand(Vec{dim, T})
                    update!(mpvalues, grid, x)
                    @test sum(mpvalues.N) ≈ 1
                    @test sum(mpvalues.∇N) ≈ zero(Vec{dim}) atol=TOL
                    l = Marble.getsupportlength(bspline)
                    if all(a->l<a<1-l, x)
                        @test grid_to_point((mp,i) -> mp.N*grid[i], mpvalues) ≈ x atol=TOL
                        @test grid_to_point((mp,i) -> grid[i]⊗mp.∇N, mpvalues) ≈ I atol=TOL
                    end
                end
            end
        end
    end
end

@testset "WLSValues" begin
    for T in (Float32, Float64)
        Random.seed!(1234)
        TOL = sqrt(eps(T))
        for dim in 1:3
            # using T=Float32 in Grid fails
            grid = Grid(ntuple(i -> 0.0:0.1:1.0, Val(dim)))
            side_length = gridsteps(grid) ./ 2
            r = Vec(side_length ./ 2)
            for kernel in (QuadraticBSpline(), CubicBSpline(), GIMP())
                for WLS in (LinearWLS, BilinearWLS)
                    WLS == BilinearWLS && dim != 2 && continue
                    mpvalues = MPValues{dim, T}(WLS(kernel))
                    for _ in 1:2000
                        x = rand(Vec{dim, T})
                        if kernel isa GIMP
                            update!(mpvalues, grid, (;x,r))
                        else
                            update!(mpvalues, grid, x)
                        end
                        @test sum(mpvalues.N) ≈ 1
                        @test sum(mpvalues.∇N) ≈ zero(Vec{dim}) atol=TOL
                        @test grid_to_point((mp,i) -> mp.N*grid[i], mpvalues) ≈ x atol=TOL
                        @test grid_to_point((mp,i) -> grid[i]⊗mp.∇N, mpvalues) ≈ I atol=TOL
                    end
                end
            end
        end
    end
end

@testset "GIMPValues" begin
    for T in (Float32, Float64)
        Random.seed!(1234)
        TOL = sqrt(eps(T))
        for dim in 1:3
            # using T=Float32 in Grid fails
            grid = Grid(ntuple(i -> 0.0:0.1:1.0, Val(dim)))
            for gimp in (GIMP(),)
                mpvalues = MPValues{dim, T}(gimp)
                side_length = gridsteps(grid) ./ 2
                r = Vec(side_length ./ 2)
                # GIMP doesn't have pertition of unity when closed to boundaries
                # if we follow eq.40 in Bardenhagen (2004)
                for _ in 1:2000
                    x = rand(Vec{dim, T})
                    if all(a->a[2]<a[1]<1-a[2], zip(x,r))
                        update!(mpvalues, grid, (;x,r))
                        @test sum(mpvalues.N) ≈ 1
                        @test sum(mpvalues.∇N) ≈ zero(Vec{dim}) atol=TOL
                        @test grid_to_point((mp,i) -> mp.N*grid[i], mpvalues) ≈ x atol=TOL
                        @test grid_to_point((mp,i) -> grid[i]⊗mp.∇N, mpvalues) ≈ I atol=TOL
                    end
                end
            end
        end
    end
end

@testset "KernelCorrectionValues" begin
    for T in (Float32, Float64)
        Random.seed!(1234)
        TOL = sqrt(eps(T))
        for dim in 1:3
            # using T=Float32 in Grid fails
            grid = Grid(ntuple(i -> 0.0:0.1:1.0, Val(dim)))
            side_length = gridsteps(grid) ./ 2
            r = Vec(side_length ./ 2)
            for kernel in (QuadraticBSpline(), CubicBSpline(), GIMP())
                mpvalues = MPValues{dim, T}(KernelCorrection(kernel))
                for _ in 1:2000
                    x = rand(Vec{dim, T}) # failed on exactly boundary
                    if kernel isa GIMP
                        update!(mpvalues, grid, (;x,r))
                    else
                        update!(mpvalues, grid, x)
                    end
                    @test sum(mpvalues.N) ≈ 1
                    @test sum(mpvalues.∇N) ≈ zero(Vec{dim}) atol=TOL
                    @test grid_to_point((mp,i) -> mp.N*grid[i], mpvalues) ≈ x atol=TOL
                    @test grid_to_point((mp,i) -> grid[i]⊗mp.∇N, mpvalues) ≈ I atol=TOL
                end
            end
        end
    end
end
