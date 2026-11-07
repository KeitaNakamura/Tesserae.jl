function test_basis_array_contracts()
    @testset "Basis-weight Base array contracts" begin
        cart = CartesianMesh(1.0,(0,2))
        weights = [generate_basis_weights(@NamedTuple{N::Float64,c::Float64},
                       BSpline(Linear()),cart,dims;deferred)
                   for dims in ((),(4,),(2,2)), deferred in (false,true)]
        cellweights = [generate_basis_weights(@NamedTuple{N::Float64,c::Float64},mesh,2,2)
                       for mesh in (FEMesh(Tesserae.Line2(),cart),IGAMesh(cart;degree=Quadratic()))]
        for w in (weights...,cellweights...)
            w.c .= 3
            ids = reshape(collect(1:length(w)),size(w))
            selection = reverse(1:length(w))
            for f in (copy, x -> view(x,first(CartesianIndices(x))),
                      x -> view(x,selection), x -> x[selection],
                      x -> copy(view(x,selection)), x -> reshape(x,1,length(x)),
                      x -> reshape(vec(x),size(x)), vec)
                actual, expected = f(w), f(ids)
                @test eltype(actual) === eltype(w)
                @test size(actual) == size(expected)
                dest = Vector{eltype(w)}(undef,length(actual))
                @test copyto!(dest,actual) === dest
                @test all(p -> dest[p] === w[expected[p]],eachindex(dest))
                first(actual).c[1] = 7
                @test w[first(expected)].c[1] == 7
                w.c .= 3
            end
            for T in (Int32,UInt32,BigInt)
                I = map(T,Tuple(first(CartesianIndices(w))))
                @test (@inferred w[I...]) === first(w)
            end
            isempty(size(w)) || @test_throws ArgumentError w[ntuple(_ -> true,ndims(w))...]
        end
        for mesh in (FEMesh(Tesserae.Line2(),cart),IGAMesh(cart;degree=Quadratic()))
            points = generate_particles(@NamedTuple{x::Vec{1,Float64},m::Float64,s::Float64},mesh,generate_quadrature_rule(basis(mesh)))
            w = generate_basis_weights(mesh,size(points))
            update!(w,points,mesh)
            points.m .= 1
            indices = reverse(1:2:length(points))
            weights, particles = view(w,indices), view(parent(points),indices)
            grid = generate_grid(@NamedTuple{x::Vec{1,Float64},m::Float64,u::Float64},mesh)
            grid.u .= first.(mesh)
            @P2G grid=>i particles=>p weights=>ip begin
                m[i] = @∑ w[ip] * m[p]
            end
            @G2P grid=>i particles=>p weights=>ip begin
                s[p] = @∑ w[ip] * u[i]
            end
            @test sum(grid.m) ≈ sum(particles.m)
            @test particles.s ≈ first.(particles.x)
        end
    end
end
