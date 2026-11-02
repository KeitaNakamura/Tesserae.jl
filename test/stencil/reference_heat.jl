using Tesserae.Stencil

function main(n=(32, 24); α=0.1, t_end=0.1)
    C = Region(Cell(), n)
    h = 1 ./ n
    temperature = [prod(sinpi, x) for x in coordinates(C, h)]
    flux = [similar(temperature, size(Region(Face(d), C))) for d in eachindex(n)]
    rate = similar(temperature, size(C))
    dtmax = 0.2/(α*sum(hd -> hd^(-2), h))
    nsteps = ceil(Int, t_end/dtmax)
    dt = t_end/nsteps

    for _ in 1:nsteps
        fill!(rate, 0)
        for (d, e) in enumerate(unitoffsets(C))
            F = Region(Face(d), C)
            I = validregion(F, (-e/2, e/2))
            J = validregion(C, (-e/2, e/2))
            L, R = boundaries(F, d)
            f, hd = flux[d], h[d]
            @views @. f[I] = -α*(temperature[I+e/2]-temperature[I-e/2])/hd
            @views @. f[L] = -α*temperature[L+e/2]/(hd/2)
            @views @. f[R] =  α*temperature[R-e/2]/(hd/2)
            @views @. rate[J] -= (f[J+e/2]-f[J-e/2])/hd
        end
        @. temperature += dt*rate
    end
    temperature
end
