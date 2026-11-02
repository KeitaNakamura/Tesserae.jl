@testset "Same gradient in face and cell coordinates" begin
    for n in ((5, 4), (4, 3, 2)), dtype in (Float32, Float64)
        N = length(n)
        C = Region(Cell(), n)
        h = ntuple(d -> dtype(0.1d), N)
        slopes = ntuple(d -> dtype(d), N)
        T = [sum(slopes[d]*(J[d]-dtype(0.5))*h[d] for d in 1:N)
             for J in CartesianIndices(n)]
        offsets = unitoffsets(C)
        for d in 1:N
            F = Region(Face(d), C)
            I = interior(F)
            e, hd = offsets[d], h[d]
            face_form = similar(T, size(Region(Face(d), C)))
            cell_form = similar(face_form)
            J = I-e/2
            @views @. face_form[I] = -(T[I+e/2]-T[I-e/2])/hd
            @views @. cell_form[J+e/2] = -(T[J+e]-T[J])/hd
            @test face_form[I] == cell_form[I]
            @test all(v -> isapprox(v, -slopes[d]; atol=32eps(dtype), rtol=0), face_form[I])
        end
    end
end

@testset "Flux balance including boundary cells" begin
    n = (7, 5)
    h = (0.2, 0.3)
    dt = 0.0001
    T = [1.0+0.1i+0.2j for i in 1:n[1], j in 1:n[2]]
    before = sum(T)*prod(h)
    rate = zero(T)
    C = Region(Cell(), n)
    offsets = unitoffsets(C)
    q = [similar(T, size(Region(Face(d), C))) for d in 1:2]
    for d in 1:2
        F = Region(Face(d), C)
        I = interior(F)
        L, R = boundaries(F, d)
        f, e, hd = q[d], offsets[d], h[d]
        @views @. f[I] = -(T[I+e/2]-T[I-e/2])/hd
        if d == 1
            @views @. f[L] = -(T[L+e/2]-2)/(hd/2)
            @views @. f[R] = -(3-T[R-e/2])/(hd/2)
        else
            fill!(view(f, L), 0.0)
            fill!(view(f, R), 0.4)
        end
        @views @. rate[C] -= (f[C+e/2]-f[C-e/2])/hd
    end
    @. T += dt*rate
    outward = sum((sum(selectdim(q[d], d, n[d]+1))-sum(selectdim(q[d], d, 1)))*
                  prod(h)/h[d] for d in 1:2)
    @test abs(sum(T)*prod(h)-before+dt*outward) < 1e-14
end

@testset "MAC pressure operator" begin
    n = (4, 3)
    h = (0.2, 0.3)
    C = Region(Cell(), n)
    offsets = unitoffsets(C)
    function pressure_operator(p)
        result = zero(p)
        for d in 1:2
            g = similar(p, size(Region(Face(d), C)))
            fill!(g, 0) # impermeable boundaries
            I = interior(Region(Face(d), C))
            e, hd = offsets[d], h[d]
            @views @. g[I] = (p[I+e/2]-p[I-e/2])/hd
            @views @. result[C] -= (g[C+e/2]-g[C-e/2])/hd
        end
        result
    end
    columns = map(1:prod(n)) do k
        p = zeros(n)
        p[k] = 1
        vec(pressure_operator(p))
    end
    A = reduce(hcat, columns)
    @test A ≈ A' atol=1e-14 rtol=0
    @test maximum(abs, A*ones(prod(n))) < 1e-13
    @test minimum(eigvals(Symmetric(A))) > -1e-13
end