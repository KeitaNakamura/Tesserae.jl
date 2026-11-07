function deferred_weight_transfers(weights, particles, grid)
    @P2G grid=>i particles=>p weights=>ip begin
        m[i] = @∑ N[ip] * m[p]
        g[i] = @∑ ∇N[ip] * m[p]
        c[i] = @∑ c[ip] * m[p]
    end
    scatter = (Array(cpu(grid.m)), Array(cpu(grid.g)), Array(cpu(grid.c)))
    @G2P grid=>i particles=>p weights=>ip begin
        s[p] = @∑ N[ip] * u[i]
        g[p] = @∑ ∇N[ip] * u[i]
    end
    gather = (Array(cpu(particles.s)), Array(cpu(particles.g)))
    @G2P2G grid=>i particles=>p weights=>ip begin
        s[p] = @∑ N[ip] * u[i]
        x[p] += Vec(0.125f0, 0.125f0)
        l[p] *= 0.5f0
        m[i] = @∑ N[ip] * m[p]
        g[i] = @∑ ∇N[ip] * m[p]
        c[i] = @∑ c[ip] * m[p]
    end
    (scatter..., gather..., Array(cpu(particles.s)), Array(cpu(grid.m)),
     Array(cpu(grid.g)), Array(cpu(grid.c)), Array(cpu(particles.x)), Array(cpu(particles.l)))
end

function deferred_weight_snapshot(weights, particles, grid)
    # No weight is read before the particle state changes.
    @G2P2G grid=>i particles=>p weights=>ip begin
        x[p] += Vec(0.125f0, 0.125f0)
        l[p] *= 0.5f0
        m[i] = @∑ N[ip] * m[p]
        g[i] = @∑ ∇N[ip] * m[p]
        c[i] = @∑ c[ip] * m[p]
    end
    (Array(cpu(grid.m)), Array(cpu(grid.g)), Array(cpu(grid.c)),
     Array(cpu(particles.x)), Array(cpu(particles.l)))
end

function test_deferred_weights(to)
    @testset "Stored and deferred weights retain custom data ($to)" begin
        for T in (to === identity ? (Float32, Float64) : (Float32,))
            V = Vec{2,T}
            mesh = CartesianMesh(T,one(T),(zero(T),T(4)),(zero(T),T(4)))
            ps = generate_particles(@NamedTuple{x::V,m::Float32,s::Float32,g::Vec{2,Float32},l::T},mesh;alg=GridSampling())[1:3]
            ps.x .= [V(T(1.125),T(1.125)),V(T(1.75),T(1.25)),V(T(2.25),T(1.75))]
            ps.m .= Float32[1,2,3]; ps.l .= T(0.5)
            grid = generate_grid(@NamedTuple{x::V,m::Float32,g::Vec{2,Float32},c::Float32,u::Float32},mesh)
            grid.u .= (x -> 1+sum(x)).(mesh)
            mask = trues(size(mesh)); mask[1,1] = false
            for b in (BSpline(Quadratic()),uGIMP(),WLS(BSpline(Quadratic())),
                      WLS(BSpline(Quadratic()),Polynomial(MultiLinear())),
                      KernelCorrection(BSpline(Quadratic())),KernelCorrection(uGIMP())),
                mode in (:stored,:flagged,:storageless)
                ref = generate_basis_weights(@NamedTuple{N::Float32,c::Float32},b,mesh,length(ps))
                w = to(generate_basis_weights(@NamedTuple{N::Float32,c::Float32},b,mesh,length(ps);deferred=mode==:storageless))
                particles, background = to(ps), to(mesh)
                indices, sub = to([3,1,2]), to([2,1])
                views = (w,view(view(w,indices),sub))
                pviews = (particles,view(view(particles,indices),sub))
                filters = Tesserae.supports_filtered_updates(b) ? (nothing,mask,nothing) : (nothing,)
                for filter in filters
                    if filter === nothing
                        update!(ref,ps,mesh)
                        update!(w,particles,background;deferred=mode!=:stored)
                    else
                        update!(ref,ps,mesh,filter)
                        update!(w,particles,background,to(filter);deferred=mode!=:stored)
                    end
                    ref.c .= 3; w.c .= 3
                    if mode == :stored
                        ref.N .= 2; ref.∇N .= Ref(Vec(4f0,4f0))
                        w.N .= 2; w.∇N .= Ref(Vec(4f0,4f0))
                    elseif mode == :flagged
                        # Deferred transfers must ignore the retained buffers.
                        w.N .= NaN32; w.∇N .= Ref(Vec(NaN32,NaN32))
                    end
                    rw = to(ref)
                    refs = (rw,view(view(rw,indices),sub))
                    for (weights,reference,pv) in zip(views,refs,pviews),
                        transfer in (deferred_weight_transfers,deferred_weight_snapshot)
                        got = transfer(weights,deepcopy(pv),to(deepcopy(grid)))
                        expected = transfer(reference,deepcopy(pv),to(deepcopy(grid)))
                        for (actual,value) in zip(got,expected); @test actual ≈ value; end
                    end
                end
            end
        end
    end
    test_deferred_value_only(to)
    test_deferred_second_derivatives(to)
    test_unused_deferred_jets(to)
end

function test_deferred_value_only(to)
    @testset "Order(0) keeps derivative-named custom data ($to)" begin
        mesh = CartesianMesh(Float32,1f0,(0f0,4f0))
        ps = generate_particles(@NamedTuple{x::Vec{1,Float32},m::Float32},mesh;alg=GridSampling())[1:1]
        ps.x .= Ref(Vec(2.25f0)); ps.m .= 1
        particles, background = to(ps), to(mesh)
        for mode in (:stored,:flagged,:storageless)
            w = to(generate_basis_weights(@NamedTuple{N::Float32,∇N::Float32},BSpline(Quadratic()),mesh,1;
                    derivative=Order(0),deferred=mode==:storageless))
            update!(w,particles,background;deferred=mode!=:stored)
            # With Order(0), ∇N is caller-owned scalar data, not a basis gradient.
            w.∇N .= 5
            grid = to(generate_grid(@NamedTuple{x::Vec{1,Float32},m::Float32,c::Float32},mesh))
            @P2G grid=>i particles=>p w=>ip begin
                m[i] = @∑ N[ip] * m[p]
                c[i] = @∑ ∇N[ip] * m[p]
            end
            @test Array(cpu(grid.m)) == Float32[0,1/32,11/16,9/32,0]
            @test Array(cpu(grid.c)) == Float32[0,5,5,5,0]
        end
    end
end

function test_deferred_second_derivatives(to)
    @testset "Stored and deferred second derivatives ($to)" begin
        for dim in 1:2, b in (BSpline(Cubic()),WLS(BSpline(Cubic())),
                WLS(BSpline(Cubic()),Polynomial(MultiLinear())),KernelCorrection(BSpline(Cubic())))
            V = Vec{dim,Float32}
            mesh = CartesianMesh(Float32,1f0,ntuple(_ -> (0f0,4f0),dim)...)
            ps = generate_particles(@NamedTuple{x::V,m::Float32},mesh;alg=GridSampling())[2:3]
            ps.m .= 1
            ref = generate_basis_weights(Float32,b,mesh,length(ps);name=Val(:N),derivative=Order(2))
            H = eltype(ref.∇²N)
            function scatter(weights)
                grid = to(generate_grid(@NamedTuple{x::V,m::Float32,g::V,h::H},mesh))
                particles = to(ps)
                @P2G grid=>i particles=>p weights=>ip begin
                    m[i] = @∑ N[ip] * m[p]
                    g[i] = @∑ ∇N[ip] * m[p]
                    h[i] = @∑ ∇²N[ip] * m[p]
                end
                (Array(cpu(grid.m)),Array(cpu(grid.g)),Array(cpu(grid.h)))
            end
            mask = trues(size(mesh)); mask[begin] = false
            filters = Tesserae.supports_filtered_updates(b) ? (Tesserae.Trues(size(mesh)),mask) : (Tesserae.Trues(size(mesh)),)
            for filter in filters, mode in (:stored,:flagged,:storageless)
                update!(ref,ps,mesh,filter)
                w = to(generate_basis_weights(Float32,b,mesh,length(ps);name=Val(:N),derivative=Order(2),deferred=mode==:storageless))
                update!(w,to(ps),to(mesh),to(filter);deferred=mode!=:stored)
                for (actual,expected) in zip(scatter(w),scatter(to(ref))); @test actual ≈ expected; end
            end
        end
    end
end

function test_unused_deferred_jets(to)
    @testset "Unreferenced deferred jets are not evaluated ($to)" begin
        mesh = CartesianMesh(Float32,1f0,(0f0,4f0))
        ps = generate_particles(@NamedTuple{x::Vec{1,Float32},m::Float32},mesh;alg=GridSampling())[3:3]
        ps.m .= 1
        b = WLS(BSpline(Quadratic()))
        grid = to(generate_grid(@NamedTuple{x::Vec{1,Float32},c::Float32},mesh))
        expected = zeros(Float32,size(mesh))
        for i in Tesserae.supportnodes(b,ps[1],mesh); expected[i] = 3; end
        for storageless in (false,true)
            w = to(generate_basis_weights(@NamedTuple{N::Float32,c::Float32},b,mesh,1;deferred=storageless))
            w.c .= 3
            # The filter would make a WLS fit singular. Only c is referenced.
            update!(w,to(ps),to(mesh),to(falses(size(mesh)));deferred=true)
            particles = to(ps)
            @P2G grid=>i particles=>p w=>ip begin
                c[i] = @∑ c[ip] * m[p]
            end
            @test Array(cpu(grid.c)) == expected
        end
    end
end
