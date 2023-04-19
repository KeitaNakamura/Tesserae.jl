import PoissonDiskSampling as PDS
import Random

const poisson_disk_sampling = PDS.generate
const Particles = StructVector

#####################
# SamplingAlgorithm #
#####################

abstract type SamplingAlgorithm end

struct GridSampling <: SamplingAlgorithm end
struct PoissonDiskSampling{RNG} <: SamplingAlgorithm
    rng::RNG
end
PoissonDiskSampling() = PoissonDiskSampling(Random.GLOBAL_RNG)

##################
# SamplingDomain #
##################

abstract type SamplingDomain end

struct BoxDomain{dim, T} <: SamplingDomain
    minmax::NTuple{dim, Tuple{T, T}}
end
BoxDomain(minmax::Tuple{T, T}...) where {T <: Real} = BoxDomain(minmax)
BoxDomain(lattice::Lattice) = BoxDomain(tuple.(Tuple(first(lattice)), Tuple(last(lattice))))

entire_volume(box::BoxDomain) = prod(x->x[2]-x[1], box.minmax)

function point_sampling(::GridSampling, box::BoxDomain{dim, T}, l::T) where {dim, T}
    r = l / 2
    minmax((xmin,xmax), r) = (xmin+r, xmax-r)
    points = Lattice(2r, minmax.(box.minmax, r)...)
    points, entire_volume(box)/length(points)
end

function point_sampling(pds::PoissonDiskSampling, box::BoxDomain{dim, T}, l::T) where {dim, T}
    # Determine minimum distance between particles for Poisson disk sampling
    # so that the number of generated particles is almost the same as the grid sampling.
    # This is empirical equation and is a bit different from previous work (https://kola.opus.hbz-nrw.de/frontdoor/deliver/index/docId/2129/file/MA_Thesis_Nilles_signed.pdf)
    points = poisson_disk_sampling(pds.rng, l/(1.376)^(1/√dim), box.minmax...)
    U = eltype(eltype(points))
    reinterpret(Vec{dim,U}, points), entire_volume(box)/length(points)
end

function point_sampling(alg::SamplingAlgorithm, box::BoxDomain{dim, T}, l) where {dim, T}
    point_sampling(alg, box, convert(T, l))
end

struct FunctionDomain{F, D <: SamplingDomain} <: SamplingDomain
    isindomain::F
    entiredomain::D
end

entire_volume(domain::FunctionDomain) = entire_volume(domain.entiredomain)

function point_sampling(alg::SamplingAlgorithm, domain::FunctionDomain, l::Real)
    points, Vₚ = point_sampling(alg, domain.entiredomain, l)
    mask = broadcast(Base.splat(domain.isindomain), points)
    view(points, mask), Vₚ
end

function SphericalDomain(centroid::Vec{dim, T}, radius::T) where {dim, T}
    minmax = ntuple(Val(dim)) do i
        c = centroid[i]
        r = radius
        (c-r, c+r)
    end
    FunctionDomain((x...) -> norm(Vec(x)-centroid) < radius, BoxDomain(minmax))
end

######################
# generate_particles #
######################

Base.@pure function infer_particles_type(::Type{ParticleState}) where {ParticleState}
    Base._return_type(StructVector{ParticleState}, Tuple{UndefInitializer, Int})
end

"""
    generate_particles(isindomain, lattice; <keyword arguments>)
    generate_particles(isindomain, ParticleState, lattice; <keyword arguments>)

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
* `spacing::Real`: particle spacing in cell along with axis. When `spacing = 0.5` (default), the total number of particles per cell for `alg=GridSampling()` becomes `2`, `4` and `8` in 1D, 2D and 3D, respectively.
* `alg::SamplingAlgorithm`: choose `PoissonDiskSampling()` (default) or `GridSampling()`.
* `system::CoordinateSystem`: use `Axisymmetric()` for axisymmetric simulations.
"""
function generate_particles end

function generate_particles(::Type{ParticleState}, points::AbstractArray{<: Vec}) where {ParticleState}
    particles = StructVector{ParticleState}(undef, length(points))
    fillzero!(particles)
    @. particles.x = points
    particles
end

function generate_particles(
        domain::SamplingDomain,
        ::Type{ParticleState},
        lattice::Lattice{dim};
        spacing::Real = 0.5,
        alg::SamplingAlgorithm = PoissonDiskSampling(),
        system::CoordinateSystem = NormalSystem(),
    ) where {ParticleState, dim}

    points, Vₚ = point_sampling(alg, domain, Marble.spacing(lattice) * spacing)
    particles = generate_particles(ParticleState, points)

    if :V in propertynames(particles)
        if dim == 2 && system isa Axisymmetric
            @. particles.V = getindex(particles.x, 1) * Vₚ
        else
            @. particles.V = Vₚ
        end
    end
    if :l in propertynames(particles)
        l = Vₚ^(1/dim)
        particles.l .= l
    end

    reorder_particles!(particles, lattice)
    particles
end
function generate_particles(isindomain::Function, ::Type{ParticleState}, lattice::Lattice; kwargs...) where {ParticleState}
    domain = FunctionDomain(isindomain, BoxDomain(lattice))
    generate_particles(domain, ParticleState, lattice; kwargs...)
end

function generate_particles(domain, lattice::Lattice{dim, T}; kwargs...) where {dim, T}
    ParticleState = minimum_particle_state(Val(dim), T)
    generate_particles(domain, ParticleState, lattice; kwargs...)
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
