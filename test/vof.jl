using Tesserae: PLIC, VOF, Vec
using Test

@testset "PLIC cuts and inverse reconstruction" begin
    for T in (Float32, Float64), N in (2, 3)
        normal = Vec(ntuple(_ -> one(T), N))
        lower, upper = -normal/2, normal/2
        # A plane at distance t along every axis from the lower corner cuts a simplex.
        for t in T.((0.25, 0.5, 0.75))
            offset = t-T(N)/2
            fraction = t^N/factorial(N)
            @test PLIC.plane_fraction(normal, offset) ≈ fraction atol=4eps(T) rtol=0
            @test PLIC.plane_offset(normal, fraction) ≈ offset atol=4eps(T) rtol=0
        end
        oblique = Vec(ntuple(d -> (-one(T))^d*d, N))
        axis = Vec(ntuple(d -> T(d == 1), N))
        for fraction in T.((0, 0.01, 0.25, 0.5, 0.8, 0.99, 1))
            offset = @inferred PLIC.plane_offset(oblique, fraction)
            @test offset isa T
            @test abs(PLIC.plane_fraction(oblique, offset)-fraction) ≤ 64eps(T)
            left = PLIC.box_volume(oblique, offset, lower, upper-axis/2)
            right = PLIC.box_volume(oblique, offset, lower+axis/2, upper)
            @test left+right ≈ fraction atol=16eps(T) rtol=0
        end
        @test_throws ArgumentError PLIC.plane_offset(normal, -eps(T))
        @test_throws ArgumentError PLIC.plane_offset(normal, one(T)+eps(T))
        @test_throws ArgumentError PLIC.plane_offset(zero(normal), zero(T))
        @test_throws ArgumentError PLIC.box_volume(normal, zero(T), upper, lower)
    end
end

@testset "VOF normals and signed donor fluxes" begin
    for T in (Float32, Float64)
        uniform = fill(T(0.1), 3, 3, 3)
        I = CartesianIndex(2, 2, 2)
        @test VOF.myc_direction(uniform, I) == Vec(one(T), zero(T), zero(T))
        @test iszero(VOF.flux(T(0.25), uniform, I, 1))
    end
    for N in (2, 3), T in (Float32, Float64), axis in 1:N
        direction = Vec(ntuple(d -> T(d == axis), N))
        field = [clamp(T(2.25-I[axis]), zero(T), one(T)) for I in CartesianIndices(ntuple(_ -> 3, N))]
        center = CartesianIndex(ntuple(_ -> 2, N))
        normal = @inferred VOF.myc_normal(field, center)
        @test maximum(abs, normal-direction) ≤ 4eps(T)
        @test VOF.myc_direction(field, center) == normal
        c = field[center]
        offset = PLIC.plane_offset(normal, c)
        for a in T.((-1, -0.5, 0, 0.5, 0.875, 1))
            expected = a ≥ 0 ? max(zero(T), c+a-one(T)) : -min(c, -a)
            @test abs(VOF.flux(a, field, center, axis)-expected) ≤ 64eps(T)
            @test VOF.flux(a, c, normal, offset, axis) == VOF.flux(a, field, center, axis)
        end
        # The whole swept slab is water even when its width is sub-ulp at the face.
        for a in (-eps(T)/8, eps(T)/8, eps(T)/6)
            @test VOF.flux(a, T(0.5), -sign(a)*direction, zero(T), axis) == a
        end
        cnear = prevfloat(one(T))
        @test VOF.flux(eps(T), cnear, direction, cnear-T(0.5), axis) == eps(T)/2
        field[center] = zero(T)
        @test maximum(abs, VOF.myc_direction(field, center)-direction) ≤ 4eps(T)
        @test_throws BoundsError VOF.myc_direction(field, first(CartesianIndices(field)))
        @test_throws ArgumentError VOF.flux(T(2), field, center, axis)
        @test_throws ArgumentError VOF.flux(T(0.25), field, center, 0)
        fill!(field, T(NaN))
        for pure in (-eps(T), zero(T), one(T), one(T)+eps(T))
            field[center] = pure
            @test VOF.flux(T(0.25), field, center, axis) == T(0.25)*pure
        end
        @test_throws ArgumentError VOF.myc_direction(field, center)
    end
    # Independent oblique reference from Basilisk's myc2d.h (column-major samples).
    samples = Vec(0.125, 0.75, 0.3125, 0.1875, 0.8125, 0.0625, 0.375, 0.75, 1.0)
    @test VOF.myc_direction(samples, Val(2)) ≈ Vec(-0.375, -0.625) atol=64eps(Float64)
    @test_throws DimensionMismatch VOF.myc_direction(samples, Val(3))
end
