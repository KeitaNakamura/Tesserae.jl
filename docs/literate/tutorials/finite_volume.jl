# # Finite-volume heat diffusion
#
# Cell-centered temperature and face-centered heat flux on the unit box.
# Zero boundary temperature and forward Euler time integration.
# The same function works in one, two, and three dimensions.

using Tesserae.Stencil

function main(n=(32, 24); α=0.1, t_end=0.1)
    # 各辺の長さが1の領域。nは各方向のセル数。
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
                ## Internal faces read the two adjacent cell centers.
                f[F] = -α*(temperature[F+e/2]-temperature[F-e/2])/hd

                ## The zero-temperature walls are half a cell away.
                f[L] = -α*temperature[L+e/2]/(hd/2)
                f[R] =  α*temperature[R-e/2]/(hd/2)

                ## Every cell reads its two bounding face fluxes.
                rate[C] -= (f[C+e/2]-f[C-e/2])/hd
            end
        end
        @. temperature += dt*rate
    end

    temperature
end

# `@stencil` combines automatic common-range selection, `@views`, and `@.`.
# Each equation updates only the candidate points where every displayed read
# exists. A block executes equations in order, choosing a range per equation.
# Here the first equation updates internal faces; the boundary equations then
# set the two wall fluxes, and the last equation updates every cell.
# Scalar bindings, allocations, and standalone helper calls are prepared outside the
# block. As with `@.`, every assignment inside the block is a dot assignment;
# it does not introduce new scalar variables.
#
# Ordinary `@views @.` with a regular `Region` remains strict. To select a
# fixed range for it, use `I = validregion(F, (-e/2, e/2))` and write the flux
# formula with `I`. `interior(F)` instead means the physical interior of `F`.

T2D = main()
T3D = main((12, 10, 8))

using Test                         #src
if @isdefined(RUN_TESTS) && RUN_TESTS #src
    @testset "Finite-volume heat diffusion" begin #src
        for n in ((9,), (32, 24), (12, 10, 8), (1, 7)) #src
            α, t_end = 0.1, 0.1 #src
            h = 1 ./ n #src
            steps = ceil(Int, t_end/(0.2/(α*sum(hd -> hd^(-2), h)))) #src
            dt = t_end/steps #src
            λ = -4sum(d -> sinpi(1/(2n[d]))^2/h[d]^2, eachindex(n)) #src
            amplitude = (1+α*dt*λ)^steps #src
            expected = [amplitude*prod(sinpi((J[d]-0.5)*h[d]) for d in eachindex(n)) #src
                        for J in CartesianIndices(n)] #src
            @test maximum(abs, main(n)-expected) < 2e-14 #src
        end #src
        errors = map(((32, 24), (64, 48))) do n #src
            exact = [exp(-2π^2*0.1*0.1)*prod(sinpi((J[d]-0.5)/n[d]) for d in 1:2) #src
                     for J in CartesianIndices(n)] #src
            maximum(abs, main(n)-exact) #src
        end #src
        @test errors[1]/errors[2] > 3.8 #src
    end #src
end #src
