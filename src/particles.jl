abstract type SamplingAlgorithm end

"""
    GridSampling(spacing = 1/2)

The generated particles are aligned with the grid using uniform spacing.
Setting `spacing = 1/η` will produce `η^dim` particles per cell, where `dim` is the problem dimension.
"""
@kwdef struct GridSampling{T} <: SamplingAlgorithm
    spacing :: T = 1/2
end

function generate_points(alg::GridSampling, mesh::CartesianMesh{dim, T}) where {dim, T}
    l = T(alg.spacing) * spacing(mesh)
    domain = tuple.(Tuple(get_xmin(mesh)), Tuple(get_xmax(mesh)))
    axis((xmin,xmax)) = (xmin+l/2):l:(xmax)
    vec(CartesianMesh(axis.(domain)...))
end

struct CellSampling{V <: Union{AbstractVector{<: Vec}, Tuple{Vararg{Vec}}}} <: SamplingAlgorithm
    qpts::V
end

cell_sampling(mesh::FEMesh) = CellSampling(generate_quadrature_rule(cellshape(mesh)).points)
cell_sampling(mesh::IGAMesh) = CellSampling(generate_quadrature_rule(igabasis(mesh)).points)

function generate_points(alg::CellSampling, mesh::Union{FEMesh, IGAMesh})
    qpts = alg.qpts
    points = Matrix{eltype(mesh)}(undef, length(qpts), ncells(mesh))
    for cell in cells(mesh)
        for (i, qpt) in enumerate(qpts)
            points[i,cell] = cell_point(mesh, cell, qpt)
        end
    end
    points
end

function cell_point(mesh::FEMesh, cell, qpt)
    indices = supportnodes(mesh, cell)
    N = value(cellshape(mesh), qpt)
    sum(N .* mesh[indices])
end

function cell_point(mesh::IGAMesh, cell, qpt)
    patch = patches(mesh, cell.patch)
    ξ = span_point(patch, cell.span, qpt)
    N, _ = iga_basis_values_and_gradients(patch, cell.span, ξ)
    indices = supportnodes(mesh, cell)
    R = geometry_basis_values(N, mesh.weights, indices)
    sum(R .* mesh[indices])
end

geometry_basis_values(N, ::Nothing, indices) = N
function geometry_basis_values(N, weights::AbstractVector, indices)
    w = weights[indices]
    W = sum(N .* w)
    map((Nᵢ, wᵢ) -> Nᵢ*wᵢ/W, N, w)
end

"""
    PoissonDiskSampling(spacing = 1/2, rng = Random.default_rng())

The particles are generated based on the Poisson disk sampling.
The `spacing` parameter is used to produce a similar number of particles as are generated with [`GridSampling`](@ref).
"""
@kwdef struct PoissonDiskSampling{T, RNG} <: SamplingAlgorithm
    spacing  :: T    = 1/2
    rng      :: RNG  = Random.default_rng()
    threaded :: Bool = rng isa Random.TaskLocalRNG ? true : false
end

# Determine minimum distance between particles for Poisson disk sampling
# so that the number of generated particles is almost the same as the grid sampling.
# This empirical equation is slightly different from a previous work (https://kola.opus.hbz-nrw.de/frontdoor/deliver/index/docId/2129/file/MA_Thesis_Nilles_signed.pdf)
function poisson_disk_sampling_minimum_distance(l::Real, domain::NTuple{dim}) where {dim}
    side_lengths = map(((xmin, xmax),) -> xmax - xmin, domain)
    coefficient = 0.938001956 / 1.42825365^(1/dim) + 0.0951151 * sum(l ./ side_lengths)
    coefficient * l
end

function generate_points(alg::PoissonDiskSampling, mesh::CartesianMesh{dim, T}) where {dim, T}
    l = T(alg.spacing) * spacing(mesh)
    domain = tuple.(Tuple(get_xmin(mesh)), Tuple(get_xmax(mesh)))
    d = poisson_disk_sampling_minimum_distance(l, domain)
    reinterpret(Vec{dim, T}, poisson_disk_sampling(alg.rng, T, d, domain...; threaded=alg.threaded))
end

function _generate_particles(::Type{ParticleProp}, points::AbstractArray{<: Vec}) where {ParticleProp}
    if !(isbitstype(ParticleProp))
        error("generate_particles: the property type of grid must be `isbitstype` type")
    end
    particles = StructArray{ParticleProp}(undef, size(points))
    fillzero!(particles)
    getx(particles) .= points
    particles
end

"""
    generate_particles([ParticleProp], mesh; alg=PoissonDiskSampling())

Generate particles across the entire `mesh` domain based on the selected `alg` algorithm.
See also [`GridSampling`](@ref) and [`PoissonDiskSampling`](@ref).

The generated `particles` is a [`StructArray`](https://github.com/JuliaArrays/StructArrays.jl)
where each element is of type `ParticleProp`. The first field of `ParticleProp` is designated for
particle positions.

```jldoctest
julia> ParticleProp = @NamedTuple begin
           x  :: Vec{2, Float64}
           m  :: Float64
           V  :: Float64
           v  :: Vec{2, Float64}
           F  :: SecondOrderTensor{2, Float64, 4}
           σ  :: SymmetricSecondOrderTensor{2, Float64, 3}
       end
@NamedTuple{x::Vec{2, Float64}, m::Float64, V::Float64, v::Vec{2, Float64}, F::Tensor{Tuple{2, 2}, Float64, 2, 4}, σ::SymmetricSecondOrderTensor{2, Float64, 3}}

julia> mesh = CartesianMesh(0.5, (0,3), (0,2));

julia> particles = generate_particles(ParticleProp, mesh; alg=GridSampling());

julia> particles.F[1] = one(eltype(particles.F));

julia> particles[1]
(x = [0.125, 0.125], m = 0.0, V = 0.0, v = [0.0, 0.0], F = [1.0 0.0; 0.0 1.0], σ = [0.0 0.0; 0.0 0.0])

julia> particles.F
96-element Vector{Tensor{Tuple{2, 2}, Float64, 2, 4}}:
 [1.0 0.0; 0.0 1.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 ⋮
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
```
"""
function generate_particles(
        ::Type{ParticleProp}, mesh::CartesianMesh;
        alg::SamplingAlgorithm=PoissonDiskSampling()) where {ParticleProp}
    points = generate_points(alg, mesh)
    particles = _generate_particles(ParticleProp, points)
    particles
end

function generate_particles(
        ::Type{ParticleProp}, mesh::Union{FEMesh, IGAMesh};
        alg::SamplingAlgorithm=cell_sampling(mesh)) where {ParticleProp}
    points = generate_points(alg, mesh)
    particles = _generate_particles(ParticleProp, points)
    particles
end

function generate_particles(mesh::CartesianMesh{dim, T}; alg::SamplingAlgorithm=PoissonDiskSampling()) where {dim, T}
    generate_particles(@NamedTuple{x::Vec{dim, T}}, mesh; alg).x
end
