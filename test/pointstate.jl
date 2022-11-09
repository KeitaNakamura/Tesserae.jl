@testset "Generating point state" begin
    @testset "generation" begin
        for random in (false, true)
            # plane strain
            grid = Grid(0:0.1:10, 0:0.1:10)
            pointstate = generate_pointstate((x,y) -> true, grid; random)
            @test sum(pointstate.V) ≈ 10*10
            @test all(pointstate) do pt
                prod(2*pt.r) ≈ pt.V
            end
            # axisymmetric
            grid = Grid(0:0.1:10, 0:0.1:10; coordinate_system = Axisymmetric())
            pointstate = generate_pointstate((x,y) -> true, grid; random)
            if random == false
                @test sum(pointstate.V) ≈ 10^2/2 * 10 # 1 radian
            else
                @test sum(pointstate.V) ≈ 10^2/2 * 10 rtol=1e-2 # 1 radian
            end
        end
    end
    @testset "copied from existing pointstate" begin
        grid = Grid(0:0.1:10, 0:0.1:10)
        pointstate_old = generate_pointstate((x,y) -> true, grid; random=true)
        pointstate_new = generate_pointstate(eltype(pointstate_old), pointstate_old)
        @test pointstate_new !== pointstate_old
        @test pointstate_new.x !== pointstate_old.x
        @test pointstate_new.V !== pointstate_old.V
        @test pointstate_new.r !== pointstate_old.r
        @test pointstate_new == pointstate_old
        @test pointstate_new.x == pointstate_old.x
        @test pointstate_new.V == pointstate_old.V
        @test pointstate_new.r == pointstate_old.r
        @test pointstate_new.index == pointstate_old.index
    end
end
