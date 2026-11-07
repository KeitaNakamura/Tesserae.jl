using Tesserae
using Metal
using SparseArrays
using Test
using Tesserae: @kernel, @index

Metal.allowscalar(false)

@kernel function read_sparse_activity!(active, spinds)
    p = @index(Global)
    @inbounds active[p] = Tesserae.isactive(spinds[p])
end

@kernel function read_sparse_values!(values, source)
    p = @index(Global)
    @inbounds values[p] = source[p]
end

include("deferred_weights.jl")
test_deferred_weights(gpu)

@testset "Empty Metal particle transfers" begin
    mesh = CartesianMesh(Float32,1f0,(0f0,4f0))
    particles = gpu(generate_particles(@NamedTuple{x::Vec{1,Float32},m::Float32,v::Float32},mesh;alg=GridSampling())[1:0])
    for deferred in (false,true), sparse in (false,true)
        grid = generate_grid(sparse ? SpArray : Array,@NamedTuple{x::Vec{1,Float32},m::Float32,v::Float32},mesh)
        sparse && update_sparsity!(grid,trues(Tesserae.nblocks(mesh)))
        grid = gpu(grid)
        weights = gpu(generate_basis_weights(Float32,BSpline(Linear()),mesh,0;deferred))
        fill!(grid.m,7)
        @P2G grid=>i particles=>p weights=>ip begin
            m[i] = @∑ w[ip] * m[p]
            v[i] = m[i] + 1f0
        end
        @test all(iszero,Array(cpu(grid.m)))
        @test all(isone,Array(cpu(grid.v)))
        fill!(grid.m,7)
        @P2G grid=>i particles=>p weights=>ip begin
            m[i] += @∑ w[ip] * m[p]
        end
        @test all(==(7f0),Array(cpu(grid.m)))
        @G2P grid=>i particles=>p weights=>ip begin
            v[p] = @∑ w[ip] * v[i]
        end
        @test isempty(particles.v)
        @G2P2G grid=>i particles=>p weights=>ip begin
            v[p] = @∑ w[ip] * v[i]
            m[i] = @∑ w[ip] * m[p]
        end
        @test all(iszero,Array(cpu(grid.m)))
        partition = gpu(Partition(mesh)); update!(partition,particles.x)
        fill!(grid.m,7)
        @test_throws "No particles assigned" @P2G grid=>i particles=>p weights=>ip partition begin
            m[i] = @∑ w[ip] * m[p]
        end
        @test all(==(7f0),Array(cpu(grid.m)))
        if !deferred
            K = gpu(fill(7f0,length(mesh),length(mesh)))
            @test_throws "target must be a sparse matrix" @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
                K[i,j] = @∑ w[ip] * w[jp] * m[p]
            end
            @test all(==(7f0),Array(cpu(K)))
        end
    end
    outside = generate_particles(@NamedTuple{x::Vec{1,Float32},m::Float32},mesh;alg=GridSampling())[1:1]
    outside.x .= Ref(Vec(-2f0)); outside.m .= 1
    outside = gpu(outside)
    partition = gpu(Partition(mesh)); update!(partition,outside.x)
    grid = gpu(generate_grid(@NamedTuple{x::Vec{1,Float32},m::Float32},mesh))
    weights = gpu(generate_basis_weights(Float32,BSpline(Linear()),mesh,1;deferred=true))
    fill!(grid.m,7)
    @test_throws "No particles assigned" @P2G grid=>i outside=>p weights=>ip partition begin
        m[i] = @∑ w[ip] * m[p]
    end
    @test all(==(7f0),Array(cpu(grid.m)))
end

@testset "Metal sparse transfers" begin
    for device in (Tesserae.MetalDevice{Tesserae.CastFloat32}(),
                   Tesserae.MetalDevice{Tesserae.PreserveEltype}())
        for A in (spzeros(Float32, 3, 3), spzeros(Float64, 3, 3),
                  spzeros(Vec{2,Float64}, 3, 3), sparsevec([1], [1.0], 3))
            @test_throws ArgumentError device(A)
        end

        blocks = create_block_sparse_matrix(Float32, BSpline(Linear()), CartesianMesh(1.0, (0,2)); ndofs=(1, 1))
        @test_throws ArgumentError device(blocks)
        @test_throws ArgumentError device(blocks[1,1])
    end
end

@testset "Metal basis-weight parent and view updates" begin
    mesh = CartesianMesh(1.0, (0,4))
    particles = generate_particles(@NamedTuple{x::Vec{1,Float64}, m::Float64}, mesh; alg=GridSampling())[1:4]
    particles.x .= [Vec(0.75), Vec(1.25), Vec(2.25), Vec(2.75)]
    particles.m .= 1:4
    leftmask = trues(size(mesh)); leftmask[1] = false
    rightmask = trues(size(mesh)); rightmask[end] = false
    gm, gp = gpu(mesh), gpu(particles)
    masks = (gpu(leftmask), gpu(rightmask))
    subset = [3,1]
    device_subset = gpu(subset)

    function scatter_view(weights, particles, grid)
        @P2G grid=>i particles=>p weights=>ip begin
            m[i] = @∑ w[ip] * m[p]
            g[i] = @∑ ∇w[ip] * m[p]
        end
        (Array(cpu(grid.m)), Array(cpu(grid.g)))
    end
    grid = generate_grid(@NamedTuple{x::Vec{1,Float64}, m::Float64, g::Vec{1,Float64}}, mesh)
    gg = gpu(grid)
    function check_view_transfer(w, p, ref, rp)
        got, expected = scatter_view(w, p, gg), scatter_view(ref, rp, grid)
        @test got[1] ≈ expected[1] rtol=1f-4 atol=1f-5
        @test got[2] ≈ expected[2] rtol=1f-4 atol=1f-5
    end

    @testset "$basis, storageless=$storageless" for basis in
            (WLS(BSpline(Quadratic())), KernelCorrection(BSpline(Quadratic()))), storageless in (false, true)
        wg = gpu(generate_basis_weights(basis, mesh, length(particles); deferred=storageless))
        ref = generate_basis_weights(basis, mesh, length(particles))
        views = ((view(wg, 1:2), view(gp, 1:2), 1:2),
                 (view(wg, device_subset), view(gp, device_subset), subset),
                 (view(view(wg, 2:4), 2:3), view(view(gp, 2:4), 2:3), 3:4))
        for (mask, hostmask) in zip(masks, (leftmask, rightmask))
            update!(wg, gp, gm, mask; deferred=true)
            update!(ref, particles, mesh, hostmask)
            indices_before = Array(cpu(Tesserae.supportnodes_storage(wg)))
            values_before = storageless ? nothing : Array(cpu(wg.w))
            for (wv, pv, inds) in views
                @test Tesserae.isdeferred(wv)
                @test Tesserae.deferred_filter(wv) === mask
                @test update!(wv, pv, gm) === wv
                check_view_transfer(wv, pv, view(ref, inds), view(particles, inds))
                @test_throws ArgumentError update!(wv, pv, gm; deferred=false)
                @test_throws ArgumentError update!(wv, pv, gm, copy(mask))
                @test Tesserae.isdeferred(wg)
                @test Tesserae.deferred_filter(wg) === mask
                @test Array(cpu(Tesserae.supportnodes_storage(wg))) == indices_before
                storageless || @test Array(cpu(wg.w)) == values_before
            end
            emptyview, emptyparticles = view(wg, 1:0), view(gp, 1:0)
            @test update!(emptyview, emptyparticles, gm) === emptyview
            @test_throws ArgumentError update!(emptyview, emptyparticles, gm; deferred=false)

            # A moved view is an independent owner and can choose another filter.
            wc, pc = cpu(first(views)[1]), cpu(first(views)[2])
            update!(wc, pc, mesh, rightmask; deferred=true)
            @test Tesserae.deferred_filter(wg) === mask
            update!(ref, particles, mesh, rightmask)
            got = scatter_view(wc, pc, cpu(gg))
            expected = scatter_view(view(ref, 1:2), view(particles, 1:2), grid)
            @test got[1] ≈ expected[1] rtol=1f-4 atol=1f-5
            @test got[2] ≈ expected[2] rtol=1f-4 atol=1f-5

            if !storageless
                update!(wg, gp, gm, mask)
                update!(ref, particles, mesh, hostmask)
                for (wv, pv, inds) in views
                    @test !Tesserae.isdeferred(wv)
                    @test update!(wv, pv, gm) === wv
                    check_view_transfer(wv, pv, view(ref, inds), view(particles, inds))
                end
                # A local filter change in stored mode must stay local.
                wv, pv, inds = views[2]
                update!(wv, pv, gm, masks[2])
                for p in inds
                    update!(ref[p], Tesserae.LazyRow(particles, p), mesh, rightmask)
                end
                @test Tesserae.deferred_filter(wg) === mask
                check_view_transfer(wg, gp, ref, particles)
            end
        end
    end
end

@testset "Metal deferred basis weight transfers" begin
    mesh = CartesianMesh(1.0, (0,4))
    particles = generate_particles(@NamedTuple{x::Vec{1,Float64}, m::Float64}, mesh; alg=GridSampling())[2:2]
    particles.m .= 1
    mask = trues(size(mesh)); mask[1] = false
    function scatter_mass!(grid, particles, weights)
        @P2G grid=>i particles=>p weights=>ip begin
            m[i] = @∑ w[ip] * m[p]
            g[i] = @∑ ∇w[ip] * m[p]
        end
        grid
    end
    for deferred in (false, true)
        grid = generate_grid(@NamedTuple{x::Vec{1,Float64}, m::Float64, g::Vec{1,Float64}}, mesh)
        weights = generate_basis_weights(KernelCorrection(BSpline(Quadratic())), mesh, 1; deferred)
        update!(weights, particles, mesh, mask; deferred=true)
        scatter_mass!(grid, particles, weights)
        gg, pg, wg = gpu(grid), gpu(particles), gpu(weights)
        @test Tesserae.isdeferred(wg)
        @test eltype(wg.w) === Float32
        @test eltype(wg.∇w) === Vec{1,Float32}
        @test Tesserae.deferring_state(wg) !== Tesserae.deferring_state(weights)
        scatter_mass!(gg, pg, wg)
        Metal.synchronize()
        @test cpu(gg.m) ≈ grid.m rtol=1f-4 atol=1f-5
        @test cpu(gg.g) ≈ grid.g rtol=1f-4 atol=1f-5
        wc, pc, gc = cpu(wg), cpu(pg), cpu(gg)
        @test Tesserae.isdeferred(wc)
        @test Tesserae.deferring_state(wc) !== Tesserae.deferring_state(wg)
        scatter_mass!(gc, pc, wc)
        @test gc.m ≈ grid.m rtol=1f-4 atol=1f-5
        @test gc.g ≈ grid.g rtol=1f-4 atol=1f-5
    end
end

@testset "Metal multidimensional tensor transfers" begin
    function scatter_tensors!(grid, particles, weights, partition)
        @P2G grid=>i particles=>p weights=>ip partition begin
            m[i] = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
            mA[i] = @∑ w[ip] * m[p] * A[p]
        end
    end
    function gather_scatter_tensors!(grid, particles, weights)
        @G2P2G grid=>i particles=>p weights=>ip begin
            v[p] = @∑ w[ip] * mv[i]
            A[p] = @∑ w[ip] * mA[i]
            f[i] = @∑ w[ip] * v[p]
            S[i] = @∑ w[ip] * A[p]
        end
    end

    @testset "dim=$dim, sparse=$sparse" for dim in 1:3, sparse in (false, true)
        V = Vec{dim,Float32}
        M = Mat{dim,dim,Float32,dim^2}
        mesh = CartesianMesh(Float32, 0.2f0, ntuple(_ -> (0f0, 1f0), dim)...)
        grid = generate_grid(sparse ? SpArray : Array,
            @NamedTuple{x::V, m::Float32, mv::V, mA::M, f::V, S::M}, mesh)
        particles = generate_particles(@NamedTuple{x::V, m::Float32, v::V, A::M}, mesh; alg=GridSampling())
        for p in eachindex(particles)
            particles.m[p] = 0.5f0 + 0.01f0 * p
            particles.v[p] = V(ntuple(d -> Float32(d) + particles.x[p][d], dim))
            particles.A[p] = M(ntuple(k -> Float32(k) + 0.01f0 * p, dim^2))
        end
        sparse && update_sparsity!(grid, particles.x)
        weights = generate_basis_weights(Float32, BSpline(Quadratic()), mesh, length(particles))
        update!(weights, particles, mesh)
        scatter_tensors!(grid, particles, weights, nothing)
        reference_particles = deepcopy(particles)
        gather_scatter_tensors!(grid, reference_particles, weights)

        for deferred in (false, true)
            gg, pg = gpu(grid), gpu(particles)
            wg = gpu(generate_basis_weights(Float32, BSpline(Quadratic()), mesh, length(particles); deferred))
            update!(wg, pg, gg.x)
            partition = gpu(Partition(mesh))
            update!(partition, pg.x)
            for schedule in (nothing, partition)
                scatter_tensors!(gg, pg, wg, schedule)
                Metal.synchronize()
                @test Array(cpu(gg.m)) ≈ Array(grid.m) rtol=1f-4
                @test Array(cpu(gg.mv)) ≈ Array(grid.mv) rtol=1f-4
                @test Array(cpu(gg.mA)) ≈ Array(grid.mA) rtol=1f-4
            end
            gather_scatter_tensors!(gg, pg, wg)
            Metal.synchronize()
            @test Array(cpu(gg.f)) ≈ Array(grid.f) rtol=1f-4
            @test Array(cpu(gg.S)) ≈ Array(grid.S) rtol=1f-4
            @test cpu(pg.v) ≈ reference_particles.v rtol=1f-4
            @test cpu(pg.A) ≈ reference_particles.A rtol=1f-4
        end
    end
end

@testset "Metal corrected basis with filters" begin
    function scatter_corrected!(grid, particles, weights)
        @P2G grid=>i particles=>p weights=>ip begin
            m[i] = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
            g[i] = @∑ ∇w[ip] * m[p]
        end
    end
    for dim in 2:3
        V = Vec{dim,Float32}
        mesh = CartesianMesh(Float32, 1f0, ntuple(_ -> (0f0,4f0), dim)...)
        particles = generate_particles(@NamedTuple{x::V,m::Float32,v::V}, mesh; alg=GridSampling())
        particles.m .= 1
        fill!(particles.v, ones(V))
        mask = trues(size(mesh)); mask[ntuple(_ -> 3, dim)...] = false
        grid = generate_grid(@NamedTuple{x::V,m::Float32,mv::V,g::V}, mesh)
        pg, gm, filter = gpu(particles), gpu(mesh), gpu(mask)
        for basis in (WLS(BSpline(Quadratic())),
                      WLS(BSpline(Quadratic()), Polynomial(MultiLinear())),
                      KernelCorrection(BSpline(Quadratic())))
            reference = generate_basis_weights(Float32, basis, mesh, length(particles))
            update!(reference, particles, mesh, mask)
            scatter_corrected!(grid, particles, reference)
            for deferred in (false,true)
                gg = gpu(grid)
                wg = gpu(generate_basis_weights(Float32, basis, mesh, length(particles); deferred))
                update!(wg, pg, gm, filter)
                scatter_corrected!(gg, pg, wg)
                Metal.synchronize()
                @test cpu(gg.m) ≈ grid.m rtol=1f-4 atol=1f-5
                @test cpu(gg.mv) ≈ grid.mv rtol=1f-4 atol=1f-5
                @test cpu(gg.g) ≈ grid.g rtol=1f-4 atol=1f-5
            end
        end
    end
end
