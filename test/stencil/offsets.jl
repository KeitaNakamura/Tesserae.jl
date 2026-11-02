using Test
using Tesserae.Stencil: validregion

@testset "Tuple, varargs, and one-sided reads" begin
    C = Region(Cell(),(5,4))
    F = Region(Face(1),C)
    ex,ey = unitoffsets(C)
    p = reshape(collect(1:20),5,4)
    @test p[validregion(C,ex)] == p[1:4,:]
    @test p[validregion(C,-ex)] == p[2:5,:]
    @test p[validregion(C,(-ex,ex))] == p[2:4,:]
    @test validregion(C,ex) == validregion(C,(ex,))
    @test validregion(C,-ex,ex) == validregion(C,(-ex,ex))
    @test validregion(C,-ex,ex,-ey,ey) == validregion(C,(-ex,ex,-ey,ey))
    @test p[validregion(C,(-ex,ex,-ey,ey))] == p[2:4,2:3]
    @test p[validregion(C,(-(ex+ey),ex+ey))] == p[2:4,2:3]
    @test p[validregion(C,ex+ey)] == p[1:4,1:3]
    @test validregion(C,(-ex/2,ex/2)) == C
    @test validregion(F,(-ex/2,ex/2)) == interior(F)
    @test validregion(F,(-ex,ex)) == interior(F)
    @test size(validregion(F,ex/2)) == (5,4)
    @test size(validregion(F,-ex/2)) == (5,4)
    @test p[validregion(F,ex/2)+ex/2] == p
    @test p[validregion(F,-ex/2)-ex/2] == p

    # An empty read list does not change the candidate region.
    @test validregion(C,()) == C
    @test validregion(F,()) == F
    @test interior(C) == C
    @test interior(C,1) == C
    @test size(interior(F)) == (4,4)
    @test interior(F,1) == interior(F)

    C1 = Region(Cell(),(5,))
    e = unitoffsets(C1)[1]
    I = validregion(C1,(-e,e))
    q = [10.0,20.0,40.0,80.0,160.0]
    difference = fill(NaN,5)
    @views @. difference[I] = q[I+e]-q[I-e]
    @test isequal(difference,[NaN,30.0,60.0,120.0,NaN])
end

@testset "Read selections against physical-coordinate bounds" begin
    for N in 1:3
        cells = ntuple(d -> 5-d,N)
        for mask in 0:(1<<N)-1
            center2 = ntuple(d -> iszero(mask & (1 << (d-1))) ? 1 : 0,N)
            counts = ntuple(d -> cells[d]+Int(center2[d] == 0),N)
            R = Region(Tesserae.Stencil.Location(UInt(mask)),counts)
            directions = unitoffsets(R)
            e = directions[1]
            lists = [(),(e,),(-e,e),(-e/2,e/2),(-2e,2e)]
            if N > 1
                f = directions[end]
                push!(lists,(-(e+f),e+f))
                push!(lists,(e/2,-f/2))
            end
            source_values = [sum(10^d*(2*(K[d]-1)+center2[d]) for d in 1:N)
                             for K in CartesianIndices(counts)]
            for offsets in lists
                I = validregion(R,offsets)
                expected = [all(0 <= 2*(K[d]-1)+center2[d]+shift.doubled[d] <= 2cells[d]
                                for shift in offsets for d in 1:N)
                            for K in CartesianIndices(counts)]
                selected = falses(counts)
                @views @. selected[I] = true
                @test selected == expected
                @test size(view(source_values,I)) == size(I)
                isempty(offsets) || @test I == validregion(R,offsets...)
                for shift in offsets
                    target_center2 = ntuple(d -> Int(isodd(center2[d]+shift.doubled[d])),N)
                    target_counts = ntuple(d -> cells[d]+Int(target_center2[d] == 0),N)
                    target_values = [sum(10^d*(2*(K[d]-1)+target_center2[d]) for d in 1:N)
                                     for K in CartesianIndices(target_counts)]
                    change = sum(10^d*shift.doubled[d] for d in 1:N)
                    @test target_values[I+shift] == source_values[I].+change
                end
            end
        end
    end
end

@testset "Read selections compose with boundaries and coordinates" begin
    C = Region(Cell(),(5,4))
    ex,ey = unitoffsets(C)
    I = validregion(C,(-ex,ex))
    p = reshape(collect(1:20),5,4)
    @test validregion(I,ex) == I
    @test interior(I) == I
    @test p[validregion(I,-2ex,2ex)] == p[3:3,:]
    h,origin = (0.5,2.0),(1.0,-1.0)
    points = collect(coordinates(C,h; origin))
    @test collect(coordinates(I,h; origin)) == points[2:4,:]
    @test_throws ArgumentError Region(Face(1),I)

    F = Region(Face(1),C)
    J = validregion(F,(-ex/2,ex/2))
    @test all(r -> size(r) == (0,4),boundaries(J))
    V = Region(Vertex(),C)
    left = first(boundaries(V,1))
    strip = validregion(left,(-ey/2,ey/2))
    @test strip == interior(left,2)
    @test size(strip) == (1,3)
    @test validregion(left,ex) == left
    @test size(validregion(left,-ex)) == (0,5)

    far = validregion(C,20ex)
    @test size(far) == (0,4)
    @test size(p[far+20ex]) == (0,4)
    narrow = Region(Cell(),(1,3))
    en = unitoffsets(narrow)[1]
    empty = validregion(narrow,(-en,en))
    q = zeros(1,3)
    @test size(empty) == (0,3)
    @test size(q[empty-en]) == (0,3)
    @test size(q[empty+en]) == (0,3)
end

@testset "Offset dimensions and legacy axis validation" begin
    C = Region(Cell(),(5,4))
    ex,ey = unitoffsets(C)
    e3 = unitoffsets(Region(Cell(),(5,4,3)))[1]
    @test_throws MethodError validregion(C,e3)
    @test_throws MethodError validregion(C,(e3,))
    @test_throws MethodError validregion(C,ex,e3)
    @test_throws MethodError validregion(C,(ex,e3))
    @test_throws MethodError validregion(C,(ex,1))
    @test_throws ArgumentError interior(C,0)
    @test_throws ArgumentError interior(C,3)
    @test_throws MethodError interior(C,ex)
    @test_throws MethodError interior(C,(-ex,ex))
end
