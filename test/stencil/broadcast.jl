using Test
using Tesserae.Stencil: common, validregion

@testset "Explicit common adapters with standard views and broadcast" begin
    C = Region(Cell(), (5,))
    e = first(unitoffsets(C))
    values = [10., 20., 40., 80., 160.]
    result = fill(-999., 5)
    source, target = common(values), common(result)
    written = @views @. target[C] = source[C+e] - source[C-e]
    @test result == [-999., 30., 60., 120., -999.]
    @test parentindices(written) == (2:4,)
    @test common(source) === source
    @test parent(view(source, 2:4)) === values
    @test source[1] == values[1]
    @test_throws BoundsError view(values, C+e)
end

@testset "Joint range rather than independently shortened views" begin
    C = Region(Cell(),(5,))
    e = unitoffsets(C)[1]
    R = C
    p = [10.,20.,40.,80.,160.]
    out = fill(-999.,5)
    written = @stencil out[R] = p[R+e]-p[R-e]
    @test out == [-999.,30.,60.,120.,-999.]
    @test parentindices(written) == (2:4,)
    @test_throws BoundsError (@views p[C+e])

    out .= -999
    @stencil out[R] = p[R+e]-p[R]
    @test out == [10.,20.,40.,80.,-999.]
    out .= -999
    @stencil out[R] = p[R]-p[R-e]
    @test out == [-999.,10.,20.,40.,80.]
    out .= -999
    @stencil out[R] = p[R+2e]-p[R-2e]
    @test out == [-999.,-999.,150.,-999.,-999.]
end

@testset "Face/cell half shifts and multidimensional expressions" begin
    for n in ((8,6),(1,4),(4,1))
        C = Region(Cell(),n)
        Fx,Fy = Region(Face(1),C),Region(Face(2),C)
        ex,ey = unitoffsets(C)
        h = 1 ./ n
        p = [cospi(x)*cospi(y) for (x,y) in coordinates(C,h)]
        u = [sinpi(x)*cospi(y) for (x,y) in coordinates(Fx,h)]
        v = [cospi(x)*sinpi(y) for (x,y) in coordinates(Fy,h)]
        gx,expected = fill(-999.,size(Fx)),fill(-999.,size(Fx))
        R = Fx
        I = validregion(Fx,(-ex/2,ex/2))
        @stencil gx[R] = (p[R+ex/2]-p[R-ex/2])/h[1]
        @views @. expected[I] = (p[I+ex/2]-p[I-ex/2])/h[1]
        @test gx == expected

        R = C
        divergence,reference = similar(p),similar(p)
        @stencil divergence[R] = (u[R+ex/2]-u[R-ex/2])/h[1] + (v[R+ey/2]-v[R-ey/2])/h[2]
        @views @. reference[C] = (u[C+ex/2]-u[C-ex/2])/h[1] + (v[C+ey/2]-v[C-ey/2])/h[2]
        @test divergence == reference
    end
    n = (7,5,4)
    C = Region(Cell(),n)
    ex,ey,ez = unitoffsets(C)
    R = C
    p = reshape(Float64.(1:prod(n)),n)
    out,reference = fill(-999.,n),fill(-999.,n)
    I = validregion(C,(-ex,ex,-ey,ey,-ez,ez))
    @stencil out[R] = (p[R+ex]-2p[R]+p[R-ex]) + (p[R+ey]-2p[R]+p[R-ey]) + (p[R+ez]-2p[R]+p[R-ez])
    @views @. reference[I] = (p[I+ex]-2p[I]+p[I-ex]) + (p[I+ey]-2p[I]+p[I-ey]) + (p[I+ez]-2p[I]+p[I-ez])
    @test out == reference
end

@testset "Periodic opposite boundaries and alias handling" begin
    C = Region(Cell(),(5,4))
    F = Region(Face(1),C)
    ex,ey = unitoffsets(C)
    L,U = boundaries(F)
    RL,RU = L,U
    p = reshape(Float64.(1:20),5,4)
    out = fill(-999.,size(F))
    @stencil out[RL] = p[RL+ex/2]-p[RU-ex/2]
    @test out[1:1,:] == p[1:1,:]-p[end:end,:]
    @test all(==(-999.),out[2:end,:])

    C1 = Region(Cell(),(5,))
    e = unitoffsets(C1)[1]
    R = C1
    q = [10.,20.,40.,80.,160.]
    @stencil q[R] = q[R+e]-q[R-e]
    @test q == [10.,30.,60.,120.,160.]
end

@testset "Ordinary arguments, scalar assignment, and shape errors" begin
    C = Region(Cell(),(5,4))
    ex,ey = unitoffsets(C)
    R = C
    p = reshape(Float64.(1:20),5,4)
    out = fill(-999.,size(p))
    coefficient = reshape(Float64.(1:5),5,1)
    @stencil out[R] = sin(p[R+ex])-p[R-ex]*coefficient
    @test out[2:4,:] == sin.(p[3:5,:])-p[1:3,:].*coefficient[2:4,:]
    @test all(==(-999.),out[[1,end],:])
    @stencil out[R] = 2
    @test all(==(2),out)
    @stencil out[R] = coefficient
    @test out == repeat(coefficient,1,4)
    out .= 2
    @stencil out[R] += p[R+ex]
    @test out[1:4,:] == 2 .+ p[2:5,:]
    @test all(==(2),out[end,:])
    wrong_shape = zeros(3,4)
    @test_throws DimensionMismatch (@stencil out[R] = wrong_shape+p[R])
    F = Region(Face(1),C)
    @test_throws DimensionMismatch (@stencil out[F] = p[F])
end

@testset "Empty common regions and ordinary tuple broadcasting" begin
    C = Region(Cell(),(5,))
    e = unitoffsets(C)[1]
    R = C
    p = [10.,20.,40.,80.,160.]
    out = fill(-999.,5)
    written = @stencil out[R] = p[R+20e]
    @test all(==(-999.),out)
    @test size(written) == (0,)
    @stencil out[R] = p[R+e]-p[R-e]+(1.,2.,3.,4.,5.)
    @test out == [-999.,32.,63.,124.,-999.]
end

@testset "Delegate to a backend's own broadcast style" begin
    C = Region(Cell(),(5,4))
    ex,ey = unitoffsets(C)
    R = C
    qdata = reshape(Float64.(1:20),5,4)
    q = TrackedArray(qdata,Ref(0))
    out = TrackedArray(fill(-999.,5,4),Ref(0))
    coefficient = TrackedArray(reshape(Float64.(1:5),5,1),Ref(0))
    source, target = common(q), common(out)
    @views @. target[R] = source[R+ex]-source[R-ex]*coefficient
    @test out.data[2:4,:] == qdata[3:5,:]-qdata[1:3,:].*coefficient.data[2:4,:]
    @test all(==(-999.),out.data[[1,end],:])
    @test out.calls[] == 1

    fill!(out.data, -999)
    @stencil out[C] = q[C+ex]-q[C-ex]*coefficient
    @test out.data[2:4,:] == qdata[3:5,:]-qdata[1:3,:].*coefficient.data[2:4,:]
    @test all(==(-999.),out.data[[1,end],:])
    @test out.calls[] == 2
end

@testset "Common ranges against a physical-coordinate oracle" begin
    for cells in ((4,), (1,), (4,3), (1,3), (3,1), (1,1), (3,2,1))
        N = length(cells)
        for mask in 0:(1<<N)-1
            phase = ntuple(d -> Int(iszero(mask & (1<<(d-1)))), N)
            samples = ntuple(d -> cells[d]+Int(iszero(phase[d])), N)
            R = Region(Tesserae.Stencil.Location(UInt(mask)), samples)
            directions = unitoffsets(R)
            e, f = first(directions), last(directions)
            shifts = (e/2, -e, (e+f)/2)
            arrays = map(shifts) do shift
                counts = ntuple(d -> cells[d]+Int(iseven(phase[d]+shift.doubled[d])), N)
                reshape(collect(1:prod(counts)), counts)
            end
            destination, a, b = arrays
            # Exercise shifted destinations and subsets, including empty interiors.
            for selection in (R, interior(R))
                result = fill(-999, size(destination))
                expected = copy(result)
                for K in CartesianIndices(samples)
                    on_boundary = any(d -> phase[d] == 0 && K[d] in (1,samples[d]), 1:N)
                    selection == interior(R) && on_boundary && continue
                    positions = map(shifts) do shift
                        ntuple(d -> 2*(K[d]-1)+phase[d]+shift.doubled[d], N)
                    end
                    all(x -> all(d -> 0 ≤ x[d] ≤ 2cells[d], 1:N), positions) || continue
                    indices = map(x -> CartesianIndex(map(y -> fld(y,2)+1, x)), positions)
                    expected[indices[1]] = a[indices[2]]-b[indices[3]]
                end
                target, left, right = selection+shifts[1], selection+shifts[2], selection+shifts[3]
                @stencil result[target] = a[left]-b[right]
                @test result == expected
            end
        end
    end
end
