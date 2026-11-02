using Test

@testset "A stencil macro wrapping common, views, and dot" begin
    C = Region(Cell(), (5,))
    F = Region(Face(1), C)
    e = unitoffsets(C)[1]
    temperature = [10., 20., 40., 80., 160.]
    flux = fill(NaN, size(F))
    h = [1.0]
    α = 0.1

    written = @stencil flux[F] =
        -α * (temperature[F+e/2] - temperature[F-e/2]) / h[1]
    @test isequal(flux, [NaN, -1., -2., -4., -8., NaN])
    @test parentindices(written) == (2:5,)
    @test_throws BoundsError view(temperature, C+e)

    # Per-equation range selection within a block, with ordinary boundary views.
    L, U = boundaries(F, 1)
    rate = fill(NaN, size(C))
    fill!(flux, NaN)
    @stencil begin
        flux[F] = -α * (temperature[F+e/2] - temperature[F-e/2]) / h[1]
        flux[L] = -α * temperature[L+e/2] / (h[1]/2)
        flux[U] =  α * temperature[U-e/2] / (h[1]/2)
        rate[C] = -(flux[C+e/2] - flux[C-e/2]) / h[1]
    end
    @test flux == [-2., -1., -2., -4., -8., 32.]
    @test rate == [-1., 1., 2., 4., -40.]

    # Normal indexing, including scalar coefficients and end, is unchanged.
    normal = zeros(5)
    @stencil normal[2:end-1] = temperature[3:end] - temperature[1:end-2]
    @test normal == [0., 30., 60., 120., 0.]
    @test_throws BoundsError (@stencil normal[1:6] = 0)
    @stencil normal[begin+1:end-1] = temperature[begin+2:end] - temperature[begin:end-2]
    @test normal == [0., 30., 60., 120., 0.]

    # Dot's ordinary escape syntax still applies to scalar function calls.
    once = Ref(0)
    coefficient() = (once[] += 1; 2.0)
    @stencil normal[C] = temperature[C] * $coefficient()
    @test normal == 2temperature
    @test once[] == 1

    # Compound update, alias handling and nested array indexing.
    q = copy(temperature)
    @stencil q[C] += q[C+e] - q[C-e]
    @test q == [10., 50., 100., 200., 160.]
    fields = [zeros(size(F)), zeros(size(F))]
    d = 2
    @stencil fields[d][F] = temperature[F+e/2] - temperature[F-e/2]
    @test fields[2] == [0., 10., 20., 40., 80., 0.]
    @test all(iszero, fields[1])
end

@testset "Block ranges, ordering, and unsupported destinations" begin
    C = Region(Cell(), (5,))
    e = unitoffsets(C)[1]
    p = [10., 20., 40., 80., 160.]
    forward, backward = fill(NaN, 5), fill(NaN, 5)
    result = zeros(5)
    written = @stencil begin
        forward[C] = p[C+e] - p[C]
        backward[C] = p[C] - p[C-e]
        result[C] = forward[C] + backward[C]
    end
    @test isequal(forward, [10., 20., 40., 80., NaN])
    @test isequal(backward, [NaN, 10., 20., 40., 80.])
    @test isequal(result, [NaN, 30., 60., 120., NaN])
    @test parentindices(written) == (1:5,)

    # An ordinary destination must not start writing deferred region inputs.
    fill!(result, -999)
    @test_throws ArgumentError (@stencil result = p[C+e])
    @test all(==(-999), result)

    # Helpers are resolved in the library, not in the caller's local scope.
    let common_source = nothing, common = nothing
        @stencil result[C] = p[C+e] - p[C-e]
    end
    @test result == [-999., 30., 60., 120., -999.]
end
