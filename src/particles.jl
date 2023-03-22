import PoissonDiskSampling
import Random

const Particles = StructVector

function grid_sampling(lattice::Lattice, n::Int)
    axes = get_axes(lattice)
    r = spacing(lattice) / 2n
    minmax(ax, r) = (first(ax)+r, last(ax)-r)
    Lattice(2r, minmax.(axes, r)...)
end

function poisson_disk_sampling(rng, lattice::Lattice{dim}, n::Int) where {dim}
    # Determine minimum distance `d` between particles for Poisson disk sampling
    # so that the number of generated particles is almost the same as the grid sampling.
    # This is empirical equation (see https://kola.opus.hbz-nrw.de/frontdoor/deliver/index/docId/2129/file/MA_Thesis_Nilles_signed.pdf)
    d = spacing(lattice) / n / (1.7)^(1/dim)
    minmaxes = map((min,max)->(min,max), Tuple(first(lattice)), Tuple(last(lattice)))
    points = PoissonDiskSampling.generate(rng, minmaxes...; r = only(unique(d)))
    map(eltype(lattice), points)
end

function point_sampling(random::Bool, lattice::Lattice, n::Int)
    if random == true
        poisson_disk_sampling(Random.GLOBAL_RNG, lattice, n)
    else
        grid_sampling(lattice, n)
    end
end
point_sampling(rng, lattice::Lattice, n::Int) = poisson_disk_sampling(rng, lattice, n)

Base.@pure function infer_particles_type(::Type{ParticleState}) where {ParticleState}
    Base._return_type(StructVector{ParticleState}, Tuple{UndefInitializer, Int})
end

function generate_particles(::Type{ParticleState}, points::AbstractArray{<: Vec}) where {ParticleState}
    particles = StructVector{ParticleState}(undef, length(points))
    fillzero!(particles)
    @. particles.x = points
    particles
end

function generate_particles(
        isindomain::Function,
        ::Type{ParticleState},
        lattice::Lattice{dim};
        n::Int = 2,
        random = false,
        system::CoordinateSystem = NormalSystem(),
    ) where {ParticleState, dim}

    allpoints = point_sampling(random, lattice, n)
    mask = broadcast(Base.splat(isindomain), allpoints)
    particles = generate_particles(ParticleState, view(allpoints, mask))

    # currently points are generated in the entire domain
    # so, simply deviding the total volume by the number of all points
    # gives the volume of a particle.
    V = prod(last(lattice) - first(lattice)) / length(allpoints)

    if :V in propertynames(particles)
        if dim == 2 && system isa Axisymmetric
            @. particles.V = getindex(particles.x, 1) * V
        else
            @. particles.V = V
        end
    end
    if :l in propertynames(particles)
        if random === false
            l = spacing(lattice) / n
        else
            l = V^(1/dim)
        end
        particles.l .= l
    end

    reorder_particles!(particles, lattice)
    particles
end

function generate_particles(isindomain::Function, lattice::Lattice{dim, T}; kwargs...) where {dim, T}
    ParticleState = minimum_particle_state(Val(dim), T)
    generate_particles(isindomain, ParticleState, lattice; kwargs...)
end

"""
    generate_particles(isindomain, grid; <keyword arguments>)
    generate_particles(isindomain, ParticleState, grid; <keyword arguments>)

Generate particles (material points) with type `ParticleState`.

`isindomain` is a function where particles are actually generated if `isindomain` returns `true`.
The arguments of `isindomain` are the coordinates. For example, `isindomain(x,y)` in 2D and
`isindomain(x,y,z)` in 3D.

This function returns `StructVector` (see [StructArrays.jl](https://github.com/JuliaArrays/StructArrays.jl)).
It is strongly recommended that `ParticleState` is bits type, i.e., `isbitstype(ParticleState)`
returns `true`. It is possible to use `NamedTuple` for `ParticleState`.

# Properties

Some property names have specific meaning and they are automatically initialized.

* `x::Vec` : particle position
* `V::Real` : particle volume
* `l::Real` : particle size

If `ParticleState` is not given, the `NamedTuple` including above properties is used.

# Keyword arguments
* `n::Int`: the number of particles in cell along with axis. So, when `n = 2` is given, the total number of particles per cell becomes `2`, `4` and `8` in 1D, 2D and 3D, respectively. `n = 2` is used by default.
* `random::Bool`: Poisson disk sampling is used when `random = true` (`random = false` by default). In the random sampling, minimum distance between particles is set to `spacing(grid) / n`.
* `system::CoordinateSystem`: use `Axisymmetric()` for axisymmetric simulations.
"""
generate_particles(isindomain::Function, ::Type{ParticleState}, grid::Grid; kwargs...) where {ParticleState} = generate_particles(isindomain, ParticleState, grid.x; kwargs...)
generate_particles(isindomain::Function, grid::Grid; kwargs...) = generate_particles(isindomain, grid.x; kwargs...)

function generate_particles(::Type{ParticleState}, particles_old::StructVector) where {ParticleState}
    particles = StructVector{ParticleState}(undef, length(particles_old))
    fillzero!(particles)

    if :x in propertynames(particles)
        particles.x .= particles_old.x
    end
    if :V in propertynames(particles)
        particles.V .= particles_old.V
    end
    if :l in propertynames(particles)
        particles.l .= particles_old.l
    end

    particles
end

function generate_particles(particles_old::StructVector)
    T = eltype(particles_old.x)
    ParticleState = minimum_particle_state(Val(length(T)), eltype(T))
    generate_particles(ParticleState, particles_old)
end

function minimum_particle_state(::Val{dim}, ::Type{T}) where {dim, T}
    @NamedTuple begin
        x::Vec{dim, T}
        V::T
        l::T
    end
end

# reorder_particles!
function _reorder_particles!(particles::Particles, ptsinblks::AbstractArray)
    inds = Vector{Int}(undef, sum(length, ptsinblks))

    cnt = 1
    for blocks in threadsafe_blocks(size(ptsinblks))
        @inbounds for blockindex in blocks
            particleindices = ptsinblks[blockindex]
            for i in eachindex(particleindices)
                inds[cnt] = particleindices[i]
                particleindices[i] = cnt
                cnt += 1
            end
        end
    end

    # keep missing particles aside
    if length(inds) != length(particles) # some points are missing
        missed = particles[setdiff(1:length(particles), inds)]
    end

    # reorder particles
    @inbounds particles[1:length(inds)] .= view(particles, inds)

    # assign missing particles to the end part of `particles`
    if length(inds) != length(particles)
        @inbounds particles[length(inds)+1:end] .= missed
    end

    particles
end

function reorder_particles!(particles::Particles, lattice::Lattice)
    xₚ = particles.x
    ptsinblks = map(_->Int[], lattice)
    @inbounds for p in eachindex(xₚ)
        I = whichblock(lattice, xₚ[p])
        I === nothing || push!(ptsinblks[I], p)
    end
    _reorder_particles!(particles, ptsinblks)
end
