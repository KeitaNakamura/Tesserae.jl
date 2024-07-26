abstract type SamplingAlgorithm end

struct GridSampling <: SamplingAlgorithm end

function point_sampling(::GridSampling, l::T, domain::Vararg{Tuple{T, T}, dim}) where {dim, T}
    axis((xmin,xmax)) = (xmin+l/2):l:(xmax)
    vec(CartesianMesh(axis.(domain)...))
end

struct PoissonDiskSampling{RNG} <: SamplingAlgorithm
    rng::RNG
    multithreading::Bool
end
PoissonDiskSampling() = PoissonDiskSampling(Random.default_rng(), true)
PoissonDiskSampling(rng; multithreading=false) = PoissonDiskSampling(rng, multithreading)

# Determine minimum distance between particles for Poisson disk sampling
# so that the number of generated particles is almost the same as the grid sampling.
# This empirical equation is slightly different from a previous work (https://kola.opus.hbz-nrw.de/frontdoor/deliver/index/docId/2129/file/MA_Thesis_Nilles_signed.pdf)
poisson_disk_sampling_minimum_distance(l::Real, dim::Int) = l/(1.37)^(1/√dim)
function point_sampling(pds::PoissonDiskSampling, l::T, domain::Vararg{Tuple{T, T}, dim}) where {dim, T}
    d = poisson_disk_sampling_minimum_distance(l, dim)
    reinterpret(Vec{dim, T}, poisson_disk_sampling(pds.rng, T, d, domain...; pds.multithreading))
end

function generate_particles(::Type{ParticleProp}, points::AbstractVector{<: Vec}) where {ParticleProp}
    _generate_particles(ParticleProp, points)
end
function _generate_particles(::Type{ParticleProp}, points::AbstractVector{<: Vec}) where {ParticleProp}
    particles = StructVector{ParticleProp}(undef, length(points))
    fillzero!(particles)
    getx(particles) .= points
    particles
end

"""
    generate_particles([ParticleProp], mesh; spacing=0.5, alg=PoissonDiskSampling())

Generate particles with particle `spacing` by sampling `alg`orithm.
"""
function generate_particles(
        ::Type{ParticleProp}, mesh::CartesianMesh{dim, T};
        spacing::Real=0.5, alg::SamplingAlgorithm=PoissonDiskSampling(), domain=nothing) where {ParticleProp, dim, T}
    points = _generate_points(ParticleProp, mesh, spacing, alg, domain)
    particles = _generate_particles(ParticleProp, points)
    _reorder_particles!(particles, mesh)
end

function generate_particles(mesh::CartesianMesh{dim, T}; spacing::Real=0.5, alg::SamplingAlgorithm=PoissonDiskSampling(), domain=nothing) where {dim, T}
    generate_particles(@NamedTuple{x::Vec{dim, T}}, mesh; spacing, alg, domain).x
end

struct Box{dim, T}
    domain::NTuple{dim, Tuple{T, T}}
end
function Box(domain::Vararg{Tuple{Real, Real}, dim}) where {dim}
    T = mapreduce(x -> promote_type(map(eltype, x)...), promote_type, domain)
    Box{dim, T}(domain)
end
isinside(x::Vec{dim}, box::Box{dim}) where {dim} = all(ntuple(i -> box.domain[i][1] ≤ x[i] < box.domain[i][2], Val(dim)))
function volume(box::Box)
    prod(box.domain) do (xmin, xmax)
        xmax - xmin
    end
end

function _generate_points(::Type{ParticleProp}, mesh::CartesianMesh{dim, T}, spacing::Real, alg::SamplingAlgorithm, ::Nothing) where {ParticleProp, dim, T}
    domain = tuple.(Tuple(first(mesh)), Tuple(last(mesh)))
    point_sampling(alg, Tesserae.spacing(mesh) * T(spacing), domain...)
end
function _generate_points(::Type{ParticleProp}, mesh::CartesianMesh{dim, T}, spacing::Real, alg::GridSampling, box::Box) where {ParticleProp, dim, T}
    domain = tuple.(Tuple(first(mesh)), Tuple(last(mesh)))
    points = point_sampling(alg, Tesserae.spacing(mesh) * T(spacing), domain...)
    filter(x -> isinside(x, box), points)
end
function _generate_points(::Type{ParticleProp}, mesh::CartesianMesh{dim, T}, spacing::Real, alg::PoissonDiskSampling, box::Box) where {ParticleProp, dim, T}
    points = point_sampling(alg, Tesserae.spacing(mesh) * T(spacing), box.domain...)
end

function _reorder_particles!(particles::AbstractVector, mesh::CartesianMesh)
    ptsinblks = map(_->Int[], CartesianIndices(blocksize(mesh)))
    for p in eachindex(particles)
        I = whichblock(getx(particles)[p], mesh)
        I === nothing || push!(ptsinblks[I], p)
    end
    reorder_particles!(particles, ptsinblks)
end
