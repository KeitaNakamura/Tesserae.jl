@testset "SpArray" begin

@testset "SpIndex" begin
    @test Tesserae.isactive(Tesserae.SpIndex(1, 0)) === false
    @test Tesserae.isactive(Tesserae.SpIndex(1, 1)) === true
    @test Tesserae.isactive(Tesserae.SpIndex(1, 2)) === true
    @test Tesserae.isactive(Tesserae.SpIndex(CartesianIndex(1,1), 0)) === false
    @test Tesserae.isactive(Tesserae.SpIndex(CartesianIndex(1,1), 1)) === true
    @test Tesserae.isactive(Tesserae.SpIndex(CartesianIndex(1,1), 2)) === true
    A = rand(3,3)
    @test (@inferred A[Tesserae.SpIndex(1,0)]) === zero(eltype(A))
    @test (@inferred A[Tesserae.SpIndex(1,1)]) === A[1]
    @test (@inferred A[Tesserae.SpIndex(CartesianIndex(1,2),0)]) === zero(eltype(A))
    @test (@inferred A[Tesserae.SpIndex(CartesianIndex(1,2),1)]) === A[1,2]
end

@testset "SpIndices" begin
    spinds = Tesserae.SpIndices(9,17)
    @test IndexStyle(spinds) === IndexCartesian()
    @test size(spinds) === (9,17)
    @test Tesserae.nblocks(spinds) === Tesserae.nblocks(size(spinds); block_size_log2=Val(Tesserae.block_size_log2(spinds)))
    @test !all(Tesserae.isactive, spinds)
    @test !all(i->Tesserae.isactive(spinds,i), eachindex(spinds))
    blkspy = rand(Bool, Tesserae.nblocks(spinds))
    n = update_sparsity!(spinds, blkspy)
    @test n == count(blkspy) * Tesserae.blocklength(spinds)
    inds = zeros(Int, size(spinds))
    for I in CartesianIndices(inds)
        block, localindex = Tesserae.global_to_blocklocal(Tuple(I)...; block_size_log2=Val(Tesserae.block_size_log2(spinds)))
        if blkspy[block...]
            linear_blkindex = LinearIndices(blkspy)[block...]
            nblks = count(blkspy[1:linear_blkindex])
            blkunit = Tesserae.blocklength(spinds)
            index = (nblks-1) * blkunit + localindex
            inds[I] = index
        end
    end
    @test map(i->i.spindex, spinds) == inds
    active = collect(Tesserae.activeindices(spinds))
    expected_active = filter(I->Tesserae.isactive(spinds, I), CartesianIndices(spinds))
    @test all(Tesserae.isactive, active)
    @test Set(map(Tesserae.logicalindex, active)) == Set(expected_active)
    @test map(Tesserae.storageindex, active) == sort(map(I->spinds[I].spindex, expected_active))
    @test all(i->1 ≤ Tesserae.storageindex(i) ≤ n, active)
    @test isempty(collect(Tesserae.activeindices(Tesserae.SpIndices(9,17))))

    update_sparsity!(spinds, falses(Tesserae.nblocks(spinds)))
    @test isempty(collect(Tesserae.activeindices(spinds)))
    @test all(i->!Tesserae.isactive(i), spinds)

    edge_spinds = Tesserae.SpIndices(5, 5)
    edge_n = update_sparsity!(edge_spinds, trues(Tesserae.nblocks(edge_spinds)))
    edge_active = collect(Tesserae.activeindices(edge_spinds))
    @test length(edge_active) == length(edge_spinds)
    @test all(i->1 ≤ Tesserae.storageindex(i) ≤ edge_n, edge_active)
    @test Set(map(Tesserae.logicalindex, edge_active)) == Set(CartesianIndices(edge_spinds))

    spinds_block3 = Tesserae.SpIndices((9, 17); block_size_log2=Val(3))
    @test Tesserae.block_size_log2(spinds_block3) === 3
    @test Tesserae.blockwidth(spinds_block3) === 8
    @test Tesserae.blocksize(spinds_block3) === (8, 8)
    @test Tesserae.nblocks(spinds_block3) === (2, 3)
    @test update_sparsity!(spinds_block3, trues(Tesserae.nblocks(spinds_block3))) ==
          prod(Tesserae.nblocks(spinds_block3)) * Tesserae.blocklength(spinds_block3)
    @test Tesserae.isactive(spinds_block3, 9, 17)
    @test_throws MethodError Tesserae.SpIndices((9, 17); block_size_log2=3)
end

@testset "SpGrid update from particles" begin
    mesh = CartesianMesh(0.25, (0,4), (0,4))
    xₚ = generate_particles(mesh)
    filter!(xₚ) do (x, y)
        (x - 2)^2 + (y - 2)^2 < 1
    end

    GridProp = @NamedTuple{x::Vec{2,Float64}, m::Float64}
    grid = generate_grid(SpArray, GridProp, mesh)
    update_sparsity!(grid, xₚ)
    from_particles = Tesserae.get_spinds(grid)
    n_particles = length(Tesserae.get_data(grid.m))

    expected_occupied = falses(Tesserae.nblocks(mesh))
    expected_counts = zeros(Int, Tesserae.nblocks(mesh))
    expected_blockids = zeros(Int, length(xₚ))
    expected_active = falses(Tesserae.nblocks(mesh))
    LI = LinearIndices(expected_occupied)
    CI = CartesianIndices(expected_active)
    for p in eachindex(xₚ)
        x = xₚ[p]
        I = Tesserae.findblock(x, mesh)
        I === nothing && continue
        expected_counts[I] += 1
        expected_blockids[p] = LI[I]
        if !expected_occupied[I]
            expected_occupied[I] = true
            blks = (I - oneunit(I)):(I + oneunit(I))
            expected_active[blks ∩ CI] .= true
        end
    end

    tracker = Tesserae.sparsity_tracker(from_particles)
    @test tracker.blockids == expected_blockids
    @test tracker.counts == expected_counts
    @test map(!iszero, Tesserae.occupied_blocks(from_particles)) == expected_occupied
    @test n_particles == count(expected_active) * Tesserae.blocklength(from_particles)
    @test map(!iszero, Tesserae.blocknumbering(from_particles)) == expected_active

    mesh = CartesianMesh(1.0, (0,16), (0,4))
    spinds = Tesserae.SpIndices(mesh)
    xₚ = [Vec(1.0, 1.0), Vec(2.0, 1.0), Vec(9.0, 1.0)]
    n₁ = update_sparsity!(spinds, xₚ, mesh)
    blocknumbering₁ = copy(Tesserae.blocknumbering(spinds))

    xₚ[1] = Vec(10.0, 1.0)
    n₂ = update_sparsity!(spinds, xₚ, mesh)
    tracker = Tesserae.sparsity_tracker(spinds)
    LI = LinearIndices(Tesserae.nblocks(mesh))
    bid₁ = LI[Tesserae.findblock(Vec(2.0, 1.0), mesh)]
    bid₂ = LI[Tesserae.findblock(Vec(10.0, 1.0), mesh)]
    @test n₂ === nothing
    @test Tesserae.blocknumbering(spinds) == blocknumbering₁
    @test tracker.counts[bid₁] == 1
    @test tracker.counts[bid₂] == 2

    update_sparsity!(spinds, falses(Tesserae.nblocks(spinds)))
    @test isempty(tracker.blockids)
    n₃ = update_sparsity!(spinds, xₚ, mesh)
    @test n₃ == n₁
    @test Tesserae.blocknumbering(spinds) == blocknumbering₁

    update_sparsity!(spinds, trues(Tesserae.nblocks(spinds)))
    @test update_sparsity!(spinds, Vec{2,Float64}[], mesh) == 0
    @test isempty(collect(Tesserae.activeindices(spinds)))
    @test all(iszero, Tesserae.blocknumbering(spinds))
end

@testset "SpArray GPU sparsity kernels on CPU backend" begin
    backend = Tesserae.CPU()

    activity = Bool[
        true  false true
        false true  false
    ]
    block_numbers = zeros(Int, size(activity))
    active_count = zeros(Int, 1)

    init_kernel = Tesserae.gpukernel_init_block_numbering!(backend)
    init_kernel(block_numbers, activity; ndrange=length(activity))
    Tesserae.synchronize(backend)
    @test block_numbers == Int.(activity)

    cumsum!(vec(block_numbers), vec(block_numbers))
    finalize_kernel = Tesserae.gpukernel_finalize_block_numbering!(backend)
    finalize_kernel(block_numbers, activity, active_count; ndrange=length(activity))
    Tesserae.synchronize(backend)

    expected_numbers = zeros(Int, size(activity))
    nactive = 0
    for i in eachindex(activity)
        if activity[i]
            nactive += 1
            expected_numbers[i] = nactive
        end
    end
    @test block_numbers == expected_numbers
    @test active_count[] == nactive

    mesh = CartesianMesh(1.0, (0,16), (0,4))
    dims = Tesserae.nblocks(mesh)
    LI = LinearIndices(dims)
    x₁ = [Vec(1.0, 1.0), Vec(2.0, 1.0), Vec(9.0, 1.0)]
    x₂ = [Vec(10.0, 1.0), Vec(2.0, 1.0), Vec(9.0, 1.0)]
    x₃ = [Vec(10.0, 1.0), Vec(14.0, 1.0), Vec(9.0, 1.0)]

    expected_tracker(xₚ) = begin
        blockids = zeros(Int, length(xₚ))
        counts = zeros(Int32, dims)
        for p in eachindex(xₚ)
            I = Tesserae.findblock(xₚ[p], mesh)
            I === nothing && continue
            bid = LI[I]
            blockids[p] = bid
            counts[bid] += 1
        end
        blockids, counts, map(!iszero, counts)
    end

    blockids = zeros(Int, length(x₁))
    counts = zeros(Int32, dims)
    occupied = falses(dims)
    changed = zeros(Int32, 1)

    update_kernel = Tesserae.gpukernel_update_particle_block_tracker!(backend)
    refresh_kernel = Tesserae.gpukernel_refresh_occupied_blocks!(backend)

    update_kernel(blockids, counts, x₁, mesh; ndrange=length(x₁))
    refresh_kernel(occupied, counts, changed; ndrange=length(occupied))
    Tesserae.synchronize(backend)
    expected_blockids, expected_counts, expected_occupied = expected_tracker(x₁)
    @test blockids == expected_blockids
    @test counts == expected_counts
    @test occupied == expected_occupied
    @test changed[] == count(expected_occupied)

    changed .= 0
    update_kernel(blockids, counts, x₂, mesh; ndrange=length(x₂))
    refresh_kernel(occupied, counts, changed; ndrange=length(occupied))
    Tesserae.synchronize(backend)
    expected_blockids, expected_counts, expected_occupied = expected_tracker(x₂)
    @test blockids == expected_blockids
    @test counts == expected_counts
    @test occupied == expected_occupied
    @test iszero(changed[])

    changed .= 0
    update_kernel(blockids, counts, x₃, mesh; ndrange=length(x₃))
    refresh_kernel(occupied, counts, changed; ndrange=length(occupied))
    Tesserae.synchronize(backend)
    expected_blockids, expected_counts, expected_occupied = expected_tracker(x₃)
    @test blockids == expected_blockids
    @test counts == expected_counts
    @test occupied == expected_occupied
    @test !iszero(changed[])

    active = falses(dims)
    expand_kernel = Tesserae.gpukernel_expand_occupied_blocks!(backend)
    expand_kernel(active, occupied; ndrange=length(occupied))
    Tesserae.synchronize(backend)

    expected_active = falses(dims)
    CI = CartesianIndices(expected_active)
    for I in CartesianIndices(occupied)
        if occupied[I]
            expected_active[((I - oneunit(I)):(I + oneunit(I))) ∩ CI] .= true
        end
    end
    @test active == expected_active

    block_numbers = zeros(Int, dims)
    active_count = zeros(Int, 1)
    init_kernel(block_numbers, active; ndrange=length(active))
    Tesserae.synchronize(backend)
    cumsum!(vec(block_numbers), vec(block_numbers))
    finalize_kernel(block_numbers, active, active_count; ndrange=length(active))
    Tesserae.synchronize(backend)

    expected_numbers = zeros(Int, dims)
    nactive = 0
    for i in eachindex(active)
        if active[i]
            nactive += 1
            expected_numbers[i] = nactive
        end
    end
    @test block_numbers == expected_numbers
    @test active_count[] == nactive

    GridProp = @NamedTuple{x::Vec{2,Float64}, m::Float64, v::Vec{2,Float64}}
    grid = generate_grid(SpArray, GridProp, mesh)
    update_sparsity!(grid, active)
    for i in Tesserae.activeindices(grid.m)
        grid.m[i] = Float64(Tesserae.storageindex(i))
    end
    bc = Broadcast.instantiate(Broadcast.broadcasted(*, grid.x, grid.m))
    copy_kernel = Tesserae.gpukernel_copyto_sp_broadcast!(backend)
    spinds = Tesserae.get_spinds(grid.v)
    ndrange = Tesserae._spindex_ndrange(spinds)
    copy_kernel(grid.v, bc, spinds; ndrange)
    Tesserae.synchronize(backend)
    @test all(i -> grid.v[i] == grid.x[i] * grid.m[i], Tesserae.activeindices(grid.v))
end

@testset "Array behavior" begin
    A = SpArray{Float64}(undef, 9, 17)
    @test IndexStyle(A) === IndexCartesian()
    @test size(A) === (9,17)
    @test !all(i->Tesserae.isactive(A,i), eachindex(A))
    @test all(iszero, A)
    blkspy = rand(Bool, Tesserae.nblocks(A))
    n = update_sparsity!(A, blkspy)
    @test length(Tesserae.get_data(A)) === n
    @test Tesserae.storedindices(A) == eachindex(Tesserae.get_data(A))
    @test Set(map(Tesserae.logicalindex, Tesserae.activeindices(A))) ==
          Set(filter(i->Tesserae.isactive(A,i), eachindex(A)))
    @test all(i->(A[i]=rand(); iszero(A[i])), filter(i->!Tesserae.isactive(A,i), eachindex(A)))
    @test all(i->(a=rand(); A[i]=a; A[i]==a), filter(i->Tesserae.isactive(A,i), eachindex(A)))
    fill!(A, 4)
    @test all(i->A[i] == 4, filter(i->Tesserae.isactive(A,i), eachindex(A)))
    @test all(i->iszero(A[i]), filter(i->!Tesserae.isactive(A,i), eachindex(A)))
    Tesserae.resize_fillzero_data!(A, length(Tesserae.get_data(A)))
    @test all(iszero, Tesserae.get_data(A))

    fillzero!(A)
    for i in Tesserae.activeindices(A)
        A[i] = Float64(Tesserae.storageindex(i))
    end
    @test all(i->A[Tesserae.logicalindex(i)] == Float64(Tesserae.storageindex(i)), Tesserae.activeindices(A))
    @test all(i->iszero(A[i]), filter(i->!Tesserae.isactive(A,i), eachindex(A)))

    B = SpArray{Float64}(undef, 9, 17)
    update_sparsity!(B, trues(Tesserae.nblocks(B)))
    B .= 2
    A .= B .+ 1
    @test all(i->A[i] == 3, filter(i->Tesserae.isactive(A,i), eachindex(A)))
    @test all(i->iszero(A[i]), filter(i->!Tesserae.isactive(A,i), eachindex(A)))

    mesh = CartesianMesh(1.0, (0,8), (0,16))
    grid = generate_grid(SpArray, @NamedTuple{x::Vec{2,Float64}, m::Float64, v::Vec{2,Float64}}, mesh)
    update_sparsity!(grid, blkspy)
    grid.m .= 2
    @. grid.v = grid.x * grid.m
    @test all(i -> grid.v[i] == grid.x[i] * grid.m[i], Tesserae.activeindices(grid.v))
    @test all(i -> iszero(grid.v[i]), filter(i -> !Tesserae.isactive(grid.v, i), eachindex(grid.v)))
    tmp = @. grid.x * grid.m
    @test tmp isa Matrix{Vec{2,Float64}}
    @test all(i -> tmp[i] == grid.x[i] * grid.m[i], eachindex(tmp))
    same_spinds = Tesserae.SpArray(fill(3.0, length(Tesserae.get_data(grid.m))), Tesserae.get_spinds(grid.m), true)
    sparse_tmp = grid.m .+ same_spinds
    @test sparse_tmp isa SpArray
    @test Tesserae.get_spinds(sparse_tmp) === Tesserae.get_spinds(grid.m)
    @test all(i -> sparse_tmp[i] == grid.m[i] + same_spinds[i], Tesserae.activeindices(sparse_tmp))
    @test grid.m .+ 1 isa Matrix{Float64}
    other_spinds = SpArray{Float64}(undef, size(grid.m))
    update_sparsity!(other_spinds, trues(Tesserae.nblocks(other_spinds)))
    @test grid.m .+ other_spinds isa Matrix{Float64}
end

@testset "@P2G" begin
    mesh = CartesianMesh(1.0, (0,2), (0,2))
    GridProp = @NamedTuple begin
        x  :: Vec{2, Float64}
        m  :: Float64
        mv :: Vec{2, Float64}
        v  :: Vec{2, Float64}
    end
    ParticleProp = @NamedTuple begin
        x :: Vec{2, Float64}
        m :: Float64
        v :: Vec{2, Float64}
    end

    dense_grid = generate_grid(GridProp, mesh)
    sp_grid = generate_grid(SpArray, GridProp, mesh)
    particles = generate_particles(ParticleProp, mesh; alg=GridSampling())
    weights = generate_basis_weights(BSpline(Linear()), mesh, length(particles))
    update!(weights, particles, mesh)

    for p in eachindex(particles)
        particles.m[p] = 1 + 0.1p
        particles.v[p] = Vec(0.2p, -0.3p)
    end

    update_sparsity!(sp_grid, particles.x)
    @test all(i->Tesserae.isactive(sp_grid, i), eachindex(sp_grid))
    @test all(x->Tesserae.get_spinds(x) === Tesserae.get_spinds(sp_grid), (sp_grid.m, sp_grid.mv))
    hybrid_grid = Tesserae.hybrid(sp_grid)
    @test hybrid_grid isa SpGrid
    @test Tesserae.get_spinds(hybrid_grid) === Tesserae.get_spinds(sp_grid)
    @test Tesserae.get_data(hybrid_grid.m) === Tesserae.get_data(sp_grid.m)
    @test eltype(supportnodes(weights[1], hybrid_grid)) <: Tesserae.SpIndex

    @P2G dense_grid=>i particles=>p weights=>ip begin
        m[i] = @∑ w[ip] * m[p]
        mv[i] = @∑ w[ip] * m[p] * v[p]
        invm = iszero(m[i]) ? zero(m[i]) : inv(m[i])
        v[i] = x[i] + mv[i] * invm
    end

    @P2G sp_grid=>i particles=>p weights=>ip begin
        m[i] = @∑ w[ip] * m[p]
        mv[i] = @∑ w[ip] * m[p] * v[p]
        invm = iszero(m[i]) ? zero(m[i]) : inv(m[i])
        v[i] = x[i] + mv[i] * invm
    end

    @test sp_grid.m ≈ dense_grid.m
    @test sp_grid.mv ≈ dense_grid.mv
    @test sp_grid.v ≈ dense_grid.v
    @test all(!iszero, map(Tesserae.storageindex, Tesserae.activeindices(sp_grid.m)))

    bad_mesh = CartesianMesh(1.0, (0,31), (0,31); block_size_log2=Val(2))
    bad_grid = generate_grid(SpArray, GridProp, bad_mesh)
    bad_particles = generate_particles(ParticleProp, bad_mesh; alg=GridSampling())
    filter!(bad_particles) do p
        p.x[1] > 28 && p.x[2] > 28
    end
    bad_weights = generate_basis_weights(BSpline(Linear()), bad_mesh, length(bad_particles))
    update!(bad_weights, bad_particles, bad_mesh)
    bad_activity = falses(Tesserae.nblocks(Tesserae.get_spinds(bad_grid)))
    bad_activity[1,1] = true
    update_sparsity!(bad_grid, bad_activity)
    bad_nodes = supportnodes(bad_weights[1], bad_grid)
    bad_node = bad_nodes[findfirst(i -> !Tesserae.isactive(i), bad_nodes)]
    @test_throws ErrorException Tesserae.p2g_write_index(bad_grid, bad_node)

    mesh_block3 = CartesianMesh(1.0, (0, 8), (0, 16); block_size_log2=Val(3))
    grid_block3 = generate_grid(SpArray, GridProp, mesh_block3)
    partition_block3 = ThreadPartition(mesh_block3)
    @test Tesserae.block_size_log2(Tesserae.get_spinds(grid_block3)) === 3
    @test Tesserae.nblocks(Tesserae.get_spinds(grid_block3)) === Tesserae.nblocks(Tesserae.strategy(partition_block3))
end

@testset "@P2G_Matrix" begin
    basis = BSpline(Linear())
    mesh = CartesianMesh(1.0, (0,31), (0,31); block_size_log2=Val(2))
    GridProp = @NamedTuple{x::Vec{2, Float64}, m::Float64}
    ParticleProp = @NamedTuple{x::Vec{2, Float64}}

    dense_grid = generate_grid(GridProp, mesh)
    sp_grid = generate_grid(SpArray, GridProp, mesh)
    particles = generate_particles(ParticleProp, mesh; alg=GridSampling())
    filter!(particles) do p
        x = p.x
        (x[1] - 2)^2 + (x[2] - 2)^2 < 2
    end
    weights = generate_basis_weights(basis, mesh, length(particles))
    update!(weights, particles, mesh)
    update_sparsity!(sp_grid, particles.x)

    @test length(collect(Tesserae.activeindices(Tesserae.get_spinds(sp_grid)))) < length(dense_grid)
    @test eltype(supportnodes(weights[1], sp_grid)) <: Tesserae.SpIndex

    A_dense = create_sparse_matrix(basis, mesh; ndofs=(2, 2))
    A_sp = create_sparse_matrix(basis, mesh; ndofs=(2, 2))
    A = A_dense
    @P2G_Matrix dense_grid=>(i,j) particles=>p weights=>(ip,jp) begin
        A[i,j] = @∑ ∇w[ip] ⊗ ∇w[jp]
    end
    A = A_sp
    @P2G_Matrix sp_grid=>(i,j) particles=>p weights=>(ip,jp) begin
        A[i,j] = @∑ ∇w[ip] ⊗ ∇w[jp]
    end
    @test A_sp ≈ A_dense

    B_dense = create_sparse_matrix(basis, mesh; ndofs=(2, 2))
    B_sp = create_sparse_matrix(basis, mesh; ndofs=(2, 2))
    B = B_dense
    @P2G_Matrix (dense_grid,dense_grid)=>(i,j) particles=>p (weights,weights)=>(ip,jp) begin
        B[i,j] = @∑ ∇w[ip] ⊗ ∇w[jp]
    end
    B = B_sp
    @P2G_Matrix (sp_grid,dense_grid)=>(i,j) particles=>p (weights,weights)=>(ip,jp) begin
        B[i,j] = @∑ ∇w[ip] ⊗ ∇w[jp]
    end
    @test B_sp ≈ B_dense
end

end
