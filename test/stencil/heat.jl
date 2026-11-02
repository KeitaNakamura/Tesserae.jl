function heat_diffusion(n=(32, 24); α=0.1, t_end=0.1)
    # Unit box; n gives the cell count along each axis.
    C = Region(Cell(), n)
    h = 1 ./ n
    temperature = [prod(sinpi, x) for x in coordinates(C, h)]

    faces = [Region(Face(d), C) for d in eachindex(n)]
    flux = [similar(temperature, size(F)) for F in faces]
    rate = similar(temperature, size(C))

    dtmax = 0.2/(α*sum(hd -> hd^(-2), h))
    nsteps = ceil(Int, t_end/dtmax)
    dt = t_end/nsteps

    for _ in 1:nsteps
        fill!(rate, 0)
        for (d, e) in enumerate(unitoffsets(C))
            F = faces[d]
            L, R = boundaries(F, d)
            f, hd = flux[d], h[d]

            @stencil begin
                # Internal faces read the two adjacent cell centers.
                f[F] = -α*(temperature[F+e/2]-temperature[F-e/2])/hd

                # The zero-temperature walls are half a cell away.
                f[L] = -α*temperature[L+e/2]/(hd/2)
                f[R] =  α*temperature[R-e/2]/(hd/2)

                # Every cell reads its two bounding face fluxes.
                rate[C] -= (f[C+e/2]-f[C-e/2])/hd
            end
        end
        @. temperature += dt*rate
    end

    temperature
end

@testset "Finite-volume heat diffusion" begin
    for n in ((9,), (32, 24), (12, 10, 8), (1, 7))
        α, t_end = 0.1, 0.1
        h = 1 ./ n
        steps = ceil(Int, t_end/(0.2/(α*sum(hd -> hd^(-2), h))))
        dt = t_end/steps
        λ = -4sum(d -> sinpi(1/(2n[d]))^2/h[d]^2, eachindex(n))
        amplitude = (1+α*dt*λ)^steps
        expected = [amplitude*prod(sinpi((J[d]-0.5)*h[d]) for d in eachindex(n))
                    for J in CartesianIndices(n)]
        @test maximum(abs, heat_diffusion(n)-expected) < 2e-14
    end
    errors = map(((32, 24), (64, 48))) do n
        exact = [exp(-2π^2*0.1*0.1)*prod(sinpi((J[d]-0.5)/n[d]) for d in 1:2)
                 for J in CartesianIndices(n)]
        maximum(abs, heat_diffusion(n)-exact)
    end
    @test errors[1]/errors[2] > 3.8
end
