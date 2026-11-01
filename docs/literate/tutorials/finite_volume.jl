# # Finite-volume heat diffusion
#
# Temperature is stored at cell centers and heat flux at face centers. We solve
# ∂T/∂t = -∇⋅q with q = -α∇T on the unit box, using zero boundary temperature
# and forward Euler time integration. The same code works in two or three dimensions.

using Tesserae.Stencil

function main(n=(32, 24); α=0.1, t_end=0.1)
    N = length(n)
    h = 1 ./ n
    x = ntuple(d -> ((1:n[d]) .- 0.5) .* h[d], N)
    T = [prod(sinpi(x[d][J[d]]) for d in 1:N) for J in CartesianIndices(n)]

    ## Face dimensions and array backend follow from the source and target placements.
    q = ntuple(d -> similar(T, Cell(), Face(d)), N)
    rate = similar(T)
    C = Region(Cell(), Val(N))
    offsets = unitoffsets(Val(N))

    dtmax = 0.2 / (α*sum(hd -> hd^(-2), h))
    nsteps = ceil(Int, t_end/dtmax)
    dt = t_end/nsteps

    for _ in 1:nsteps
        fill!(rate, 0)
        for d in 1:N
            F = Region(Face(d), Val(N))
            I = interior(F)
            L, R = boundaries(F, d)
            f, e, hd = q[d], offsets[d], h[d]

            ## Internal fluxes use the two cells half a cell away from each face.
            @views @. f[I] = -α*(T[I+e/2] - T[I-e/2])/hd

            ## Each boundary face is half a cell away from its adjacent cell center.
            @views @. f[L] = -α*T[L+e/2]/(hd/2)
            @views @. f[R] =  α*T[R-e/2]/(hd/2)

            ## Update every cell, including the cells touching the boundary.
            @views @. rate[C] -= (f[C+e/2] - f[C-e/2])/hd
        end
        @. T += dt*rate
    end

    T
end

# `Region` represents a placement and a selection, not an array. Concrete
# ranges are resolved when indexing an array; `@views` produces ordinary views,
# and `@.` uses Julia's standard broadcast. There is no separate execution API.
#
# `interior` preserves placement and excludes sampling points on the boundary.
# Thus `interior(C)` contains all cell centers, whereas `interior(F)` excludes
# the two boundary faces normal to `d`. Arrays have no halo in this example.
#
# The flux formula can also be written with its index anchored at the lower
# cell. With `J = I-e/2`, the exact same operation is
# `@views @. f[J+e/2] = -α*(T[J+e]-T[J])/hd`.
# Both forms use the same region and displacement arithmetic.

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
